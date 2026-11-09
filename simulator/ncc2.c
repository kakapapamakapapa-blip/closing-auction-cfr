/* Nasdaq Closing Cross simulator + outcome-sampling MCCFR (v2: scalable).
 * Same game as ncc.c (real Closing Cross timeline; see README), plus:
 *  - incremental NOII imbalance (recomputed only when the book or the midpoint changes),
 *  - multithreaded Hogwild training on shared tables (THREADS env), CAS insertion,
 *  - table sizes set by env (LOGT_A, LOGT_L, LOGT_BR),
 *  - richer abstractions (windows of recent NOII publications, exact imbalance),
 *  - training with checkpoints: evaluation and learned best responses at each checkpoint,
 *  - cross-play tournament between saved strategies.
 * Build: gcc -O3 -march=native -pthread -o ncc2 ncc2.c -lm */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>

/* ---------------- parameters ---------------- */
#define NE 330
#define MAXC 12
#define NA 5
#define MAXO 128
#define PR 24               /* price histogram half-range */
static int C = 6, Q = 2, LPMAX = 2, D = 10, CONTBUY = 0; /* CONTBUY: agent may buy in the continuous book at mid+1 */
static double LAM = 3.0, ALPHA = 0.0035, BETA = 0.05, EPS = 0.6;
static int decEpoch[NE + 1], isDec[NE];
static int tsec(int e) { return e < 30 ? 10 * e : 300 + (e - 30); }
static void make_schedule(int d) {
  memset(isDec, 0, sizeof isDec); int n = 0;
  if (d >= NE) { for (int e = 0; e < NE; e++) decEpoch[n++] = e; }
  else {
    int must[3] = {29, 209, NE - 1}, base = d - 3; if (base < 1) base = 1;
    for (int k = 0; k < base; k++) decEpoch[n++] = (int)floor((double)k * (NE - 1) / base);
    for (int k = 0; k < 3; k++) decEpoch[n++] = must[k];
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (decEpoch[j] < decEpoch[i]) { int t = decEpoch[i]; decEpoch[i] = decEpoch[j]; decEpoch[j] = t; }
    int m = 0; for (int i = 0; i < n; i++) if (m == 0 || decEpoch[m - 1] != decEpoch[i]) decEpoch[m++] = decEpoch[i];
    n = m;
  }
  D = n; for (int i = 0; i < n; i++) isDec[decEpoch[i]] = i + 1;
}

/* ---------------- RNG / hashing ---------------- */
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
typedef struct { uint64_t s[4]; } Rng;
static inline uint64_t rnext(Rng *r) {
  uint64_t *rs = r->s, x = rotl(rs[1] * 5, 7) * 9, t = rs[1] << 17;
  rs[2] ^= rs[0]; rs[3] ^= rs[1]; rs[1] ^= rs[2]; rs[0] ^= rs[3]; rs[2] ^= t; rs[3] = rotl(rs[3], 45); return x;
}
static inline double U01(Rng *r) { return (rnext(r) >> 11) * 0x1.0p-53; }
static void rseed(Rng *r, uint64_t s) {
  for (int i = 0; i < 4; i++) { s += 0x9E3779B97F4A7C15ULL; uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; r->s[i] = z ^ (z >> 31); }
}
static inline uint64_t mix(uint64_t h, uint64_t v) { h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2); h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 31; return h; }

/* ---------------- orders, book, cross ---------------- */
enum { MOC = 0, LOC = 1, IO = 2, CONT = 3 };
typedef struct { signed char side, type, owner, active; short limit; short stamp; } Order;
#define WIN 8
typedef struct {
  int theta, m, nU, nsig; signed char sig[MAXC];
  Order o[MAXO]; int no, stampc;
  int mocB, mocS; int limB[2 * PR + 1], limS[2 * PR + 1]; int imb_dirty, imb_cache;
  int a_rest, a_stamps[8], a_post[3], a_cont; double a_contcost; uint64_t a_prehist, a_obs, a_postacts, a_allobs; int a_previmb;
  int l_sell, l_buy; uint64_t l_obs, l_acts; int l_previmb;
  int imb, imbb; int win_n; unsigned char win[WIN]; /* recent publications (at decision points) */
} State;
static inline int clampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
static void add_order(State *s, int side, int type, int owner, int limit) {
  if (s->no >= MAXO) return;
  Order *o = &s->o[s->no++]; o->side = side; o->type = type; o->owner = owner; o->active = 1; o->limit = limit; o->stamp = s->stampc++;
  if (type == CONT) return;
  if (type == MOC) { if (side > 0) s->mocB++; else s->mocS++; }
  else { int li = clampi(limit, -PR, PR) + PR; if (side > 0) s->limB[li]++; else s->limS[li]++; }
  s->imb_dirty = 1;
}
static void deactivate(State *s, int i) {
  Order *o = &s->o[i]; if (!o->active) return; o->active = 0;
  if (o->type == MOC) { if (o->side > 0) s->mocB--; else s->mocS--; }
  else if (o->type != CONT) { int li = clampi(o->limit, -PR, PR) + PR; if (o->side > 0) s->limB[li]--; else s->limS[li]--; }
  s->imb_dirty = 1;
}
static int imbalance(State *s) {
  if (!s->imb_dirty) return s->imb_cache;
  int b = s->mocB, se = s->mocS, mi = clampi(s->m, -PR, PR) + PR;
  for (int i = mi; i <= 2 * PR; i++) b += s->limB[i];
  for (int i = 0; i <= mi; i++) se += s->limS[i];
  s->imb_cache = b - se; s->imb_dirty = 0; return s->imb_cache;
}
static inline int buy_exec(Order *o, int x) { return o->active && o->side > 0 && (o->type == MOC || o->limit >= x); }
static inline int sell_exec(Order *o, int x) { return o->active && o->side < 0 && (o->type == MOC || o->limit <= x); }
static int cross(State *s, int *fills) {
  int base = s->no;
  for (int k = 1; k <= 6; k++) { add_order(s, -1, CONT, 0, s->m + k); add_order(s, +1, CONT, 0, s->m - k); }
  int bestx = s->m, bestV = -1, bestI = 1 << 30, bestd = 1 << 30;
  for (int x = s->m - 7; x <= s->m + 7; x++) {
    int B = 0, S = 0;
    for (int i = 0; i < s->no; i++) { B += buy_exec(&s->o[i], x); S += sell_exec(&s->o[i], x); }
    int V = B < S ? B : S, I = abs(B - S), d = abs(x - s->m);
    if (V > bestV || (V == bestV && (I < bestI || (I == bestI && d < bestd)))) { bestV = V; bestI = I; bestd = d; bestx = x; }
  }
  int x = bestx;
  for (int i = 0; i < s->no; i++) fills[i] = 0;
  for (int side = -1; side <= 1; side += 2) {
    int rem = bestV, idx[MAXO + 16], n = 0;
    for (int i = 0; i < s->no; i++) if (side > 0 ? buy_exec(&s->o[i], x) : sell_exec(&s->o[i], x)) idx[n++] = i;
    for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) {
      Order *p = &s->o[idx[a]], *q = &s->o[idx[b]];
      int pp = p->type == MOC ? 0 : 1, qq = q->type == MOC ? 0 : 1, swap = 0;
      if (qq < pp) swap = 1;
      else if (qq == pp) {
        if (pp == 1) { int pl = side > 0 ? p->limit : -p->limit, ql = side > 0 ? q->limit : -q->limit;
          if (ql > pl || (ql == pl && q->stamp < p->stamp)) swap = 1; }
        else if (q->stamp < p->stamp) swap = 1;
      }
      if (swap) { int t = idx[a]; idx[a] = idx[b]; idx[b] = t; }
    }
    for (int a = 0; a < n && rem > 0; a++) { fills[idx[a]] = 1; rem--; }
  }
  s->no = base;
  return x;
}

/* ---------------- shared tables (Hogwild, CAS insert) ---------------- */
typedef struct { uint64_t key; float r[NA], sm[NA], bl[NA]; } Entry;
typedef struct { Entry *e; uint64_t mask; long n, cap, overflow; int logsz; } Table;
static void tinit(Table *t, int logsz) {
  t->logsz = logsz; t->mask = (1ULL << logsz) - 1; t->e = calloc(t->mask + 1, sizeof(Entry));
  if (!t->e) { fprintf(stderr, "out of memory for table 2^%d\n", logsz); exit(2); }
  t->n = 0; t->cap = (long)((t->mask + 1) * 0.75); t->overflow = 0;
}
static void tfree(Table *t) { free(t->e); t->e = NULL; }
static Entry *tget(Table *t, uint64_t key, int create) {
  if (key == 0) key = 1;
  uint64_t i = key & t->mask;
  for (;;) {
    Entry *e = &t->e[i]; uint64_t k = __atomic_load_n(&e->key, __ATOMIC_ACQUIRE);
    if (k == key) return e;
    if (k == 0) {
      if (!create) return NULL;
      if (__atomic_load_n(&t->n, __ATOMIC_RELAXED) >= t->cap) { __atomic_fetch_add(&t->overflow, 1, __ATOMIC_RELAXED); return NULL; }
      uint64_t z = 0;
      if (__atomic_compare_exchange_n(&e->key, &z, key, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) { __atomic_fetch_add(&t->n, 1, __ATOMIC_RELAXED); return e; }
      if (z == key) return e;
    }
    i = (i + 1) & t->mask;
  }
}
static void rm_policy(Entry *e, int mask, double *p) {
  double z = 0; int nl = 0;
  for (int a = 0; a < NA; a++) if (mask >> a & 1) { nl++; if (e && e->r[a] > 0) z += e->r[a]; }
  for (int a = 0; a < NA; a++) p[a] = (mask >> a & 1) ? (z > 0 ? (e->r[a] > 0 ? e->r[a] / z : 0) : 1.0 / nl) : 0;
}
static int avg_policy(Entry *e, int mask, double *p) {
  double z = 0; int nl = 0;
  for (int a = 0; a < NA; a++) if (mask >> a & 1) { nl++; if (e) z += e->sm[a]; }
  for (int a = 0; a < NA; a++) p[a] = (mask >> a & 1) ? (z > 0 ? e->sm[a] / z : 1.0 / nl) : 0;
  return z > 0;
}

/* ---------------- abstractions ----------------
   0: decision index + own state
   1: + number of U signals + current imbalance bucket (-3..3)
   2: + imbalance bucket at the previous own decision + midpoint offset (-2..2)
   3: + last 4 publications seen at own decisions (imbalance -3..3, mid -3..3) + full signal sequence
   4: + exact imbalance (-8..8), exact midpoint (-6..6), last 8 publications at own decisions
   5: perfect recall at own decision points with phi_H (own pre-cutoff action sequence -> resting set)
   6: full perfect recall at own decision points (incl. own pre-cutoff action sequence) */
static uint64_t window_hash(State *s, int k) { uint64_t h = 77; int n = s->win_n < k ? s->win_n : k; for (int i = 0; i < n; i++) h = mix(h, s->win[(s->win_n - 1 - i) % WIN]); return h; }
static uint64_t common_key(State *s, int var, uint64_t h, int pre) {
  if (var == 0) return h;
  h = mix(h, 50 + s->nU); if (!pre) h = mix(h, 200 + s->imbb);
  if (var == 1) return h;
  if (!pre) h = mix(h, 400 + clampi(s->m, -2, 2));
  if (var == 2) return h;
  for (int i = 0; i < s->nsig; i++) h = mix(h, 500 + s->sig[i]);
  if (!pre) h = mix(h, window_hash(s, var == 3 ? 4 : 8));
  if (var == 3) return h;
  if (!pre) { h = mix(h, 600 + clampi(s->imb, -8, 8)); h = mix(h, 700 + clampi(s->m, -6, 6)); }
  return h;
}
static uint64_t agent_key(State *s, int var, int pre, int k) {
  uint64_t h = mix(1, pre); h = mix(h, k);
  for (int i = 0; i < s->a_rest; i++) h = mix(h, 100 + s->a_stamps[i]);
  h = mix(h, s->a_post[0] * 64 + s->a_post[1] * 8 + s->a_post[2] + 512 * s->a_cont);
  if (var == 7) return mix(h, 900 + s->theta);                 /* oracle: knows the hidden flow type */
  if (var == 8) return pre ? h : mix(h, 200 + s->imbb);        /* only the current imbalance */
  if (var == 9) { h = mix(h, 900 + s->theta); return pre ? h : mix(h, 600 + clampi(s->imb, -8, 8)); } /* oracle + exact imbalance */
  if (var <= 4) { if (var >= 2 && !pre) h = mix(h, 300 + s->a_previmb); return common_key(s, var, h, pre); }
  for (int i = 0; i < s->nsig; i++) h = mix(h, 500 + s->sig[i]);
  h = mix(h, s->a_obs); h = mix(h, s->a_postacts);
  if (var == 5) return h;
  return mix(h, s->a_prehist);
}
static uint64_t lp_key(State *s, int var, int k) {
  uint64_t h = mix(7, k); h = mix(h, s->l_sell * 8 + s->l_buy);
  if (var <= 4) { if (var >= 2) h = mix(h, 300 + s->l_previmb); return common_key(s, var, h, 0); }
  for (int i = 0; i < s->nsig; i++) h = mix(h, 500 + s->sig[i]);
  h = mix(h, s->l_obs); h = mix(h, s->l_acts); return h;
}

/* ---------------- policies, workers ---------------- */
typedef struct { int kind; Table *t; int var; int use_avg; int heur; } Pol; /* kind 0 table, 1 heuristic */
static int heur_agent(int h, State *s, int pre, int k, int mask) {
  if (h == 1) return (pre && (mask & 2)) ? 1 : 0;
  if (h == 3) { if (pre) return 0; int e = decEpoch[k];
    if (e == 29 && s->imb <= 0 && (mask & 2)) return 1; if (e == 209 && (mask & 4)) return 2; if (e == NE - 1 && (mask & 8)) return 3; }
  return 0;
}
static int heur_lp(int h, State *s, int k, int mask) {
  if (h == 1 && decEpoch[k] == NE - 1) { if (s->imb >= 2 && (mask & 2)) return 1; if (s->imb <= -2 && (mask & 4)) return 2; }
  return 0;
}
typedef struct { int player; Entry *e; int mask; double pol[NA], spol[NA]; int a; double oppr, sampr; } Node;
typedef struct {
  int train, learn[3], adv; Pol pol[3];
  volatile long it_counter; long iters; int players_mask; double iterw_den;
} Ctx;
typedef struct {
  Rng rng; State S; Node nodes[2 * (MAXC + NE) + 8]; int nn; int trav; double iterw;
  double reach[3], samp; long unseen[3], decisions[3];
  double a1, b1, a2, b2, fill; long n;
} W;

static int choose(Ctx *c, W *w, State *s, int player, int pre, int k, int mask) {
  Pol *P = &c->pol[player]; double pol[NA], spol[NA]; Entry *e = NULL; int a;
  if (!(mask & (mask - 1))) return __builtin_ctz(mask); /* single legal action: no decision */
  w->decisions[player]++;
  if (P->kind == 1) {
    a = player == 1 ? heur_agent(P->heur, s, pre, k, mask) : heur_lp(P->heur, s, k, mask);
    if (!(mask >> a & 1)) a = 0;
    for (int b = 0; b < NA; b++) pol[b] = b == a; memcpy(spol, pol, sizeof pol);
  } else {
    uint64_t key = player == 1 ? agent_key(s, P->var, pre, k) : lp_key(s, P->var, k);
    int learning = c->train && c->learn[player];
    e = tget(P->t, key, learning);
    if (learning) rm_policy(e, mask, pol);
    else if (P->use_avg) { if (!avg_policy(e, mask, pol)) w->unseen[player]++; }
    else { if (!e) w->unseen[player]++; rm_policy(e, mask, pol); }
    if (c->train && w->trav == player) { int nl = __builtin_popcount(mask);
      for (int b = 0; b < NA; b++) spol[b] = (mask >> b & 1) ? EPS / nl + (1 - EPS) * pol[b] : 0; }
    else memcpy(spol, pol, sizeof pol);
    double u = U01(&w->rng), acc = 0; a = -1;
    for (int b = 0; b < NA; b++) if (mask >> b & 1) { acc += spol[b]; if (u < acc) { a = b; break; } }
    if (a < 0) for (int b = NA - 1; b >= 0; b--) if (mask >> b & 1) { a = b; break; }
  }
  if (c->train && P->kind == 0 && c->learn[player]) {
    Node *nd = &w->nodes[w->nn++]; nd->player = player; nd->e = e; nd->mask = mask; memcpy(nd->pol, pol, sizeof pol); memcpy(nd->spol, spol, sizeof spol);
    nd->a = a; nd->oppr = w->reach[3 - player]; nd->sampr = w->samp;
    if (w->trav != player && e) { double wt = w->iterw * w->reach[player] / w->samp; for (int b = 0; b < NA; b++) e->sm[b] += (float)(wt * pol[b]); }
  }
  w->reach[player] *= pol[a]; w->samp *= spol[a];
  return a;
}

static void episode(Ctx *c, W *w, double *u, int *outq) {
  State *s = &w->S; memset(s, 0, sizeof *s); s->imb_dirty = 1;
  Rng *R = &w->rng; w->nn = 0; w->reach[0] = w->reach[1] = w->reach[2] = 1; w->samp = 1;
  double u0 = U01(R); s->theta = u0 < 0.35 ? 0 : (u0 < 0.65 ? 1 : 2);
  static const double pbuy[3] = {0.45, 0.25, 0.10}, psell[3] = {0.10, 0.25, 0.45}, pU[3] = {0.65, 0.5, 0.35};
  s->a_prehist = 11; s->a_obs = 13; s->a_postacts = 17; s->a_allobs = 19; s->l_obs = 23; s->l_acts = 29; s->a_previmb = 3; s->l_previmb = 3;
  for (int t = 0; t < C; t++) {
    int sg = U01(R) < pU[s->theta]; s->sig[s->nsig++] = sg; s->nU += sg;
    int left = Q - (s->a_rest + s->a_post[0] + s->a_post[1] + s->a_post[2] + s->a_cont);
    int mask = 1; if (left > 0) mask |= 2; if (s->a_rest > 0) mask |= 4; if (CONTBUY && left > 0) mask |= 16;
    int a = choose(c, w, s, 1, 1, t, mask);
    if (a == 4) { s->a_cont++; s->a_contcost += s->m + 1; }
    if (a == 1) { add_order(s, +1, MOC, 1, 99); s->a_stamps[s->a_rest++] = t; }
    else if (a == 2) { for (int i = s->no - 1; i >= 0; i--) if (s->o[i].owner == 1 && s->o[i].active && s->o[i].type == MOC) { deactivate(s, i); break; } s->a_rest--; }
    s->a_prehist = mix(s->a_prehist, t * 4 + a);
    if (U01(R) < pbuy[s->theta]) add_order(s, +1, MOC, 0, 99);
    if (U01(R) < psell[s->theta]) add_order(s, -1, MOC, 0, -99);
  }
  s->m = 0; s->imb_dirty = 1;
  for (int e = 0; e < NE; e++) {
    int ts = tsec(e); double dt = e < 30 ? 10.0 : 1.0;
    int imb0 = imbalance(s);
    if (U01(R) < 0.004 * dt) { int dir = U01(R) < 0.5 + 0.15 * (imb0 > 0) - 0.15 * (imb0 < 0) ? 1 : -1; s->m += dir; s->imb_dirty = 1; }
    if (ts < 300) { if (U01(R) < 0.004 * dt * (s->theta == 0 ? 2 : 1)) add_order(s, +1, MOC, 0, 99);
                    if (U01(R) < 0.004 * dt * (s->theta == 2 ? 2 : 1)) add_order(s, -1, MOC, 0, -99); }
    if (imb0 != 0 && U01(R) < ALPHA * dt * abs(imb0)) add_order(s, imb0 > 0 ? -1 : +1, IO, 0, s->m);
    s->imb = imbalance(s); s->imbb = clampi(s->imb, -3, 3) + 3;
    uint64_t ob = s->imbb * 16 + (s->m + 8);
    s->a_allobs = mix(s->a_allobs, ob);
    if (isDec[e]) {
      int k = isDec[e] - 1;
      s->win[s->win_n % WIN] = (unsigned char)(s->imbb * 8 + clampi(s->m, -3, 3) + 3); s->win_n++;
      int mask = 1;
      if (s->a_rest + s->a_post[0] + s->a_post[1] + s->a_post[2] + s->a_cont < Q) { if (ts < 300) mask |= 2; if (ts < 480) mask |= 4; mask |= 8; if (CONTBUY) mask |= 16; }
      s->a_obs = mix(s->a_obs, ob);
      int a = choose(c, w, s, 1, 0, k, mask);
      if (a == 1) { add_order(s, +1, MOC, 1, 99); s->a_post[0]++; }
      if (a == 2) { add_order(s, +1, LOC, 1, s->m + 1); s->a_post[1]++; }
      if (a == 3) { add_order(s, +1, IO, 1, s->m); s->a_post[2]++; }
      if (a == 4) { s->a_cont++; s->a_contcost += s->m + 1; }
      s->a_postacts = mix(s->a_postacts, k * 4 + a); s->a_previmb = s->imbb;
      int lm = 1; if (s->l_sell < LPMAX) lm |= 2; if (s->l_buy < LPMAX) lm |= 4;
      s->l_obs = mix(s->l_obs, ob);
      int b = choose(c, w, s, 2, 0, k, lm);
      if (b == 1) { add_order(s, -1, IO, 2, s->m + 1); s->l_sell++; }
      if (b == 2) { add_order(s, +1, IO, 2, s->m - 1); s->l_buy++; }
      s->l_acts = mix(s->l_acts, k * 4 + b); s->l_previmb = s->imbb;
    }
  }
  int fills[MAXO + 16], v = s->m, p = cross(s, fills), q1 = 0, sold = 0, bought = 0;
  for (int i = 0; i < s->no; i++) if (fills[i]) { if (s->o[i].owner == 1) q1++; if (s->o[i].owner == 2) { if (s->o[i].side < 0) sold++; else bought++; } }
  u[1] = -(double)p * q1 - s->a_contcost - LAM * (Q - q1 - s->a_cont);
  u[2] = (double)sold * (p - v) + (double)bought * (v - p);
  if (outq) { outq[0] = q1 + s->a_cont; outq[1] = p; }
}

static void update(W *w, double util) {
  double value = util;
  for (int i = w->nn - 1; i >= 0; i--) {
    Node *nd = &w->nodes[i];
    if (nd->player != w->trav) continue;
    if (!nd->e) { value = value * nd->pol[nd->a] / nd->spol[nd->a]; continue; }
    Entry *e = nd->e; double uh[NA], vest = 0;
    for (int b = 0; b < NA; b++) if (nd->mask >> b & 1) { double bl = e->bl[b]; uh[b] = bl + (b == nd->a ? (value - bl) / nd->spol[nd->a] : 0); vest += nd->pol[b] * uh[b]; }
    double wt = nd->oppr / nd->sampr;
    for (int b = 0; b < NA; b++) if (nd->mask >> b & 1) { float nr = e->r[b] + (float)(wt * (uh[b] - vest)); e->r[b] = nr > 0 ? nr : 0; }
    e->bl[nd->a] += (float)(BETA * (value - e->bl[nd->a]));
    value = vest;
  }
}

/* ---------------- parallel drivers ---------------- */
static int THREADS = 2;
typedef struct { Ctx *c; W *w; long n; int eval; uint64_t seed; } Job;
static void *worker(void *arg) {
  Job *j = arg; Ctx *c = j->c; W *w = j->w;
  for (long i = 0; i < j->n; i++) {
    double u[3]; int q[2];
    if (j->eval) { episode(c, w, u, q); w->a1 += u[1]; w->b1 += u[1] * u[1]; w->a2 += u[2]; w->b2 += u[2] * u[2]; w->fill += q[0]; w->n++; continue; }
    long it = __atomic_add_fetch(&c->it_counter, 1, __ATOMIC_RELAXED);
    w->trav = c->players_mask == 3 ? (int)(it & 1) + 1 : c->players_mask;
    w->iterw = (double)it / c->iterw_den;
    episode(c, w, u, NULL); if (c->adv && w->trav == 2) u[2] = -u[1]; update(w, u[w->trav]);
  }
  return NULL;
}
static uint64_t g_seed = 1;
static void run_parallel(Ctx *c, long n, int eval, W *agg) {
  pthread_t th[64]; Job jobs[64]; W *ws = calloc(THREADS, sizeof(W));
  for (int t = 0; t < THREADS; t++) {
    rseed(&ws[t].rng, g_seed = g_seed * 6364136223846793005ULL + 1442695040888963407ULL + t);
    jobs[t] = (Job){c, &ws[t], n / THREADS + (t < n % THREADS), eval, 0};
    pthread_create(&th[t], NULL, worker, &jobs[t]);
  }
  for (int t = 0; t < THREADS; t++) pthread_join(th[t], NULL);
  if (agg) { memset(agg, 0, sizeof(W));
    for (int t = 0; t < THREADS; t++) { agg->a1 += ws[t].a1; agg->b1 += ws[t].b1; agg->a2 += ws[t].a2; agg->b2 += ws[t].b2; agg->fill += ws[t].fill; agg->n += ws[t].n;
      for (int p = 1; p <= 2; p++) { agg->unseen[p] += ws[t].unseen[p]; agg->decisions[p] += ws[t].decisions[p]; } } }
  free(ws);
}
/* train continues from c->it_counter up to `upto`; linear averaging with weight it/1e6 */
static void train_to(Ctx *c, long upto, int players_mask) {
  c->train = 1; c->players_mask = players_mask; c->iterw_den = 1e6;
  long n = upto - c->it_counter; if (n > 0) run_parallel(c, n, 0, NULL);
  c->train = 0;
}
typedef struct { double m1, s1, m2, fill, unseenA, unseenL; } EvalRes;
static EvalRes evaluate(Ctx *c, long n) {
  static W agg; c->train = 0; run_parallel(c, n, 1, &agg);
  EvalRes r; r.m1 = agg.a1 / agg.n; r.s1 = sqrt((agg.b1 / agg.n - r.m1 * r.m1) / agg.n); r.m2 = agg.a2 / agg.n; r.fill = agg.fill / agg.n / Q;
  r.unseenA = agg.decisions[1] ? (double)agg.unseen[1] / agg.decisions[1] : 0; r.unseenL = agg.decisions[2] ? (double)agg.unseen[2] / agg.decisions[2] : 0;
  return r;
}
static int LOGT_BR = 23;
/* learned best response of `player` against the fixed other player; better of the key variants in vmask */
static EvalRes br_eval(Ctx *base, int player, int adv, long brit, long neval, int vmask) {
  EvalRes best; int have = 0;
  for (int v = 0; v <= 4; v++) if (vmask >> v & 1) {
    Table T; tinit(&T, LOGT_BR);
    Ctx cb = *base; cb.pol[player] = (Pol){0, &T, v, 0, 0}; cb.learn[1] = cb.learn[2] = 0; cb.learn[player] = 1; cb.adv = adv; cb.it_counter = 0;
    train_to(&cb, brit, player); cb.adv = 0;
    EvalRes r = evaluate(&cb, neval);
    double score = player == 1 ? r.m1 : (adv ? -r.m1 : r.m2), bscore = !have ? -1e18 : (player == 1 ? best.m1 : (adv ? -best.m1 : best.m2));
    if (score > bscore) { best = r; have = 1; }
    tfree(&T);
  }
  return best;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int envi(const char *k, int d) { const char *v = getenv(k); return v ? atoi(v) : d; }

/* save / load average strategy tables (only entries with nonzero average) */
static void tsave(Table *t, const char *fn) {
  FILE *f = fopen(fn, "wb"); if (!f) return; long n = 0;
  for (uint64_t i = 0; i <= t->mask; i++) if (t->e[i].key) n++;
  fwrite(&n, sizeof n, 1, f);
  for (uint64_t i = 0; i <= t->mask; i++) if (t->e[i].key) { fwrite(&t->e[i].key, 8, 1, f); fwrite(t->e[i].sm, sizeof(float), NA, f); }
  fclose(f);
}
static int tload(Table *t, const char *fn) {
  FILE *f = fopen(fn, "rb"); if (!f) return 0; long n;
  if (fread(&n, sizeof n, 1, f) != 1) { fclose(f); return 0; }
  int lg = 10; while ((1L << lg) * 0.6 < n) lg++; tinit(t, lg);
  for (long i = 0; i < n; i++) { uint64_t k; float sm[NA]; if (fread(&k, 8, 1, f) != 1 || fread(sm, sizeof(float), NA, f) != NA) break; Entry *e = tget(t, k, 1); memcpy(e->sm, sm, sizeof sm); }
  fclose(f); return 1;
}

int main(int argc, char **argv) {
  THREADS = envi("THREADS", 2); LOGT_BR = envi("LOGT_BR", 23); CONTBUY = envi("CONTBUY", 0);
  if (getenv("LAM")) LAM = atof(getenv("LAM"));
  if (argc < 2) { fprintf(stderr, "modes: speed D | run VA VL D SEED CKPTS(comma) BRITERS NEVAL BRVMASK SAVEPREFIX | cross D NEVAL A1 L1 [A2 L2 ...]\n"); return 1; }
  if (!strcmp(argv[1], "speed")) {
    make_schedule(atoi(argv[2])); Table TA, TL; tinit(&TA, 22); tinit(&TL, 22);
    Ctx c; memset(&c, 0, sizeof c); c.pol[1] = (Pol){0, &TA, 2, 1, 0}; c.pol[2] = (Pol){0, &TL, 2, 1, 0}; c.learn[1] = c.learn[2] = 1;
    double t0 = now(); train_to(&c, 400000, 3); double t1 = now();
    printf("{\"D\":%d,\"threads\":%d,\"us_per_iteration_wall\":%.3f}\n", D, THREADS, (t1 - t0) / 400000 * 1e6);
    return 0;
  }
  if (!strcmp(argv[1], "voi")) {
    /* value of information: single-agent learning of the agent against a FIXED LP (heuristic h, 0 = withhold)
       with key variant VAR; reports the learned policy's cost. voi VAR D ITERS LPHEUR NEVAL */
    int v = atoi(argv[2]); make_schedule(atoi(argv[3])); long it = (long)atof(argv[4]); int h = atoi(argv[5]); long neval = (long)atof(argv[6]);
    g_seed = 7; Table T; tinit(&T, envi("LOGT_A", 24));
    Ctx c; memset(&c, 0, sizeof c); c.pol[1] = (Pol){0, &T, v, 0, 0}; c.pol[2] = (Pol){1, NULL, 0, 0, h}; c.learn[1] = 1;
    static Table TLfix; if (getenv("LPFILE")) { if (!tload(&TLfix, getenv("LPFILE"))) { fprintf(stderr, "no LP file\n"); return 2; } c.pol[2] = (Pol){0, &TLfix, envi("LPVAR", 0), 1, 0}; h = -1; }
    long ck = it / 8 > 1000000 ? it / 8 : it; double tt = 0;
    for (long upto = ck; upto <= it; upto *= 2) { if (upto > it) upto = it; double t0 = now(); train_to(&c, upto, 1); tt += now() - t0;
      c.learn[1] = 0; EvalRes r = evaluate(&c, neval); c.learn[1] = 1;
      printf("{\"contbuy\":%d,\"voi\":1,\"var\":%d,\"D\":%d,\"lp_heur\":%d,\"iters\":%ld,\"sec\":%.1f,\"entries\":%ld,\"overflow\":%ld,\"u1\":%.5f,\"u1_se\":%.5f,\"fill\":%.4f,\"unseen\":%.4f}\n",
             CONTBUY, v, D, h, upto, tt, T.n, T.overflow, r.m1, r.s1, r.fill, r.unseenA); fflush(stdout);
      if (upto == it) break; }
    return 0;
  }
  if (!strcmp(argv[1], "run")) {
    int va = atoi(argv[2]), vl = atoi(argv[3]); make_schedule(atoi(argv[4])); g_seed = atoll(argv[5]);
    long ck[32]; int nck = 0; char *tok = strtok(argv[6], ","); while (tok && nck < 32) { ck[nck++] = (long)atof(tok); tok = strtok(NULL, ","); }
    long brit = (long)atof(argv[7]), neval = (long)atof(argv[8]); int vmask = atoi(argv[9]); const char *pref = argc > 10 ? argv[10] : NULL;
    Table TA, TL; tinit(&TA, envi("LOGT_A", 24)); tinit(&TL, envi("LOGT_L", 22));
    Ctx c; memset(&c, 0, sizeof c); c.pol[1] = (Pol){0, &TA, va, 1, 0}; c.pol[2] = (Pol){0, &TL, vl, 1, 0};
    double ttrain = 0;
    for (int i = 0; i < nck; i++) {
      c.learn[1] = c.learn[2] = 1; double t0 = now(); train_to(&c, ck[i], 3); ttrain += now() - t0; c.learn[1] = c.learn[2] = 0;
      double t1 = now();
      EvalRes r = evaluate(&c, neval);
      Ctx ch = c; ch.pol[2] = (Pol){1, NULL, 0, 0, 1}; EvalRes rh = evaluate(&ch, neval);
      Ctx cw = c; cw.pol[2] = (Pol){1, NULL, 0, 0, 0}; EvalRes rw = evaluate(&cw, neval);
      Ctx cx = c; cx.pol[1] = (Pol){1, NULL, 0, 0, 1}; EvalRes rx = evaluate(&cx, neval);
      int dobr = (i == nck - 1) || getenv("BR_ALL");
      double gl = 0, ga = 0;
      if (dobr) { EvalRes rb = br_eval(&c, 2, 0, brit, neval, vmask), ra = br_eval(&c, 1, 0, brit, neval, vmask);
        gl = rb.m2 - r.m2; ga = ra.m1 - r.m1; }
      double lb = (gl > 0 ? gl : 0) + (ga > 0 ? ga : 0);
      printf("{\"contbuy\":%d,\"var_agent\":%d,\"var_lp\":%d,\"D\":%d,\"lam\":%.2f,\"threads\":%d,\"iters\":%ld,\"train_sec\":%.1f,\"eval_sec\":%.1f,"
             "\"entries_agent\":%ld,\"entries_lp\":%ld,\"cap_agent\":%ld,\"overflow\":%ld,\"u1\":%.5f,\"u1_se\":%.5f,\"u2\":%.5f,\"fill\":%.4f,"
             "\"unseen_agent\":%.4f,\"unseen_lp\":%.4f,\"u1_vs_heurLP\":%.5f,\"u1_vs_withhold\":%.5f,\"u1_heur1_vs_LP\":%.5f,"
             "\"br_done\":%d,\"gain_lp\":%.5f,\"gain_agent\":%.5f,\"nashconv_lb\":%.5f,\"br_iters\":%ld,\"br_vmask\":%d}\n",
             CONTBUY, va, vl, D, LAM, THREADS, ck[i], ttrain, now() - t1, TA.n, TL.n, TA.cap, TA.overflow + TL.overflow, r.m1, r.s1, r.m2, r.fill,
             r.unseenA, r.unseenL, rh.m1, rw.m1, rx.m1, dobr, gl, ga, lb, brit, vmask);
      fflush(stdout);
    }
    if (pref) { char fn[512]; snprintf(fn, sizeof fn, "%s_A.bin", pref); tsave(&TA, fn); snprintf(fn, sizeof fn, "%s_L.bin", pref); tsave(&TL, fn); }
    return 0;
  }
  if (!strcmp(argv[1], "cross")) {
    /* cross D NEVAL prefA:varA prefL:varL ... : every saved agent against every saved LP */
    make_schedule(atoi(argv[2])); long neval = (long)atof(argv[3]); int n = argc - 4;
    Table A[16], L[16]; int va[16]; char nm[16][256];
    int vlp[16];
    for (int i = 0; i < n; i++) { char *p = argv[4 + i], *col = strchr(p, ':'); *col = 0; va[i] = atoi(col + 1); char *c2 = strchr(col + 1, ':'); vlp[i] = c2 ? atoi(c2 + 1) : va[i]; snprintf(nm[i], 256, "%s", p);
      char fn[512]; snprintf(fn, sizeof fn, "%s_A.bin", p); tload(&A[i], fn); snprintf(fn, sizeof fn, "%s_L.bin", p); tload(&L[i], fn); }
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
      Ctx c; memset(&c, 0, sizeof c); c.pol[1] = (Pol){0, &A[i], va[i], 1, 0}; c.pol[2] = (Pol){0, &L[j], vlp[j], 1, 0};
      EvalRes r = evaluate(&c, neval);
      printf("{\"agent\":\"%s\",\"lp\":\"%s\",\"u1\":%.5f,\"u1_se\":%.5f,\"u2\":%.5f}\n", nm[i], nm[j], r.m1, r.s1, r.m2); fflush(stdout);
    }
    for (int j = 0; j < n; j++) { Ctx c; memset(&c, 0, sizeof c); c.pol[1] = (Pol){1, NULL, 0, 0, 1}; c.pol[2] = (Pol){0, &L[j], vlp[j], 1, 0};
      EvalRes r = evaluate(&c, neval); printf("{\"agent\":\"heur_MOC_early\",\"lp\":\"%s\",\"u1\":%.5f,\"u1_se\":%.5f,\"u2\":%.5f}\n", nm[j], r.m1, r.s1, r.m2); }
    return 0;
  }
  return 1;
}
