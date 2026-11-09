/* Nasdaq Closing Cross simulator + outcome-sampling MCCFR on abstractions.
 * Real timeline: dark window before 15:50 (MOC submit/cancel, hidden), NOII every 10 s 15:50-15:55,
 * every 1 s 15:55-16:00 (330 publications), MOC entry until 15:55, LOC until 15:58, IO until 16:00,
 * no cancellation after 15:50, cross at 16:00.
 * Players: 1 = execution agent (must buy Q lots), 2 = strategic liquidity provider (IO orders).
 * Exogenous: index MOC flow (hidden type theta), late MOC, offsetting liquidity attracted by the
 * published imbalance, continuous book depth, mid drift.
 * Usage: see main(). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

/* ---------------- RNG ---------------- */
static uint64_t rs[4];
static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t rnext(void) {
  uint64_t r = rotl(rs[1] * 5, 7) * 9, t = rs[1] << 17;
  rs[2] ^= rs[0]; rs[3] ^= rs[1]; rs[1] ^= rs[2]; rs[0] ^= rs[3]; rs[2] ^= t; rs[3] = rotl(rs[3], 45);
  return r;
}
static inline double U01(void) { return (rnext() >> 11) * 0x1.0p-53; }
static void rseed(uint64_t s) {
  for (int i = 0; i < 4; i++) { s += 0x9E3779B97F4A7C15ULL; uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; rs[i] = z ^ (z >> 31); }
}
static inline uint64_t mix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
  h *= 0xBF58476D1CE4E5B9ULL; h ^= h >> 31; return h;
}

/* ---------------- parameters ---------------- */
#define NE 330          /* NOII publications 15:50-16:00 */
#define MAXC 12
#define NA 4
static int C = 6;       /* pre-cutoff decision periods (15:20..15:45 every 5 min) */
static int Q = 2;       /* agent lots */
static double LAM = 3.0;/* penalty per unfilled lot, ticks */
static int LPMAX = 2;   /* LP lots per side */
static double ALPHA = 0.0035; /* offset arrival rate per epoch per lot of imbalance */
static int D = 10;      /* post-cutoff decision points */
static int decEpoch[NE + 1]; static int isDec[NE];

static int tsec(int e) { return e < 30 ? 10 * e : 300 + (e - 30); }

static void make_schedule(int d) {
  D = d; memset(isDec, 0, sizeof isDec);
  int n = 0;
  if (d >= NE) { for (int e = 0; e < NE; e++) decEpoch[n++] = e; }
  else {
    /* evenly spaced in epochs, always including the last epoch before each cutoff and the last epoch */
    int must[3] = {29, 209, NE - 1};
    int base = d - 3; if (base < 1) base = 1;
    for (int k = 0; k < base; k++) { int e = (int)floor((double)k * (NE - 1) / base); decEpoch[n++] = e; }
    for (int k = 0; k < 3; k++) decEpoch[n++] = must[k];
    /* sort & unique */
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (decEpoch[j] < decEpoch[i]) { int t = decEpoch[i]; decEpoch[i] = decEpoch[j]; decEpoch[j] = t; }
    int m = 0; for (int i = 0; i < n; i++) if (m == 0 || decEpoch[m - 1] != decEpoch[i]) decEpoch[m++] = decEpoch[i];
    n = m;
  }
  D = n; for (int i = 0; i < n; i++) isDec[decEpoch[i]] = i + 1;
}

/* ---------------- orders and cross ---------------- */
enum { MOC = 0, LOC = 1, IO = 2, CONT = 3 };
typedef struct { signed char side, type, owner, active; short limit; short stamp; } Order; /* side +1 buy, -1 sell; owner 0 exo,1 agent,2 LP */
#define MAXO 96
typedef struct {
  int theta, m, nU;
  int nsig; signed char sig[MAXC];
  Order o[MAXO]; int no;
  int stampc;
  /* agent */
  int a_rest;             /* resting pre-cutoff MOC lots */
  int a_stamps[4];
  int a_post[3];          /* committed post-cutoff MOC/LOC/IO */
  uint64_t a_prehist;     /* hash of own pre-cutoff action sequence */
  uint64_t a_obs;         /* hash of observations at own decision points */
  uint64_t a_postacts;
  uint64_t a_allobs;      /* hash of all 330 observations */
  int a_previmb;
  /* LP */
  int l_sell, l_buy;
  uint64_t l_obs, l_acts; int l_previmb;
  /* noii */
  int imb, imbb;          /* current imbalance and bucket */
} State;

static int committed_agent(State *s) { return s->a_rest + s->a_post[0] + s->a_post[1] + s->a_post[2]; }

static void add_order(State *s, int side, int type, int owner, int limit) {
  if (s->no >= MAXO) return;
  Order *o = &s->o[s->no++]; o->side = side; o->type = type; o->owner = owner; o->active = 1; o->limit = limit; o->stamp = s->stampc++;
}
/* buy interest executable at price x */
static inline int buy_exec(Order *o, int x) { return o->active && o->side > 0 && (o->type == MOC || o->limit >= x); }
static inline int sell_exec(Order *o, int x) { return o->active && o->side < 0 && (o->type == MOC || o->limit <= x); }

/* NOII imbalance at reference price m: on-close buy - sell interest (MOC, LOC, IO), no continuous book */
static int imbalance(State *s) {
  int b = 0, se = 0;
  for (int i = 0; i < s->no; i++) { Order *o = &s->o[i]; if (!o->active || o->type == CONT) continue;
    if (buy_exec(o, s->m)) b++; if (sell_exec(o, s->m)) se++; }
  return b - se;
}
static int imb_bucket(int x) { if (x > 3) x = 3; if (x < -3) x = -3; return x + 3; } /* 0..6 */

/* cross at 16:00: continuous book adds 1 lot per tick on both sides around m */
static int cross(State *s, int *fills /* per order */) {
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
  /* allocate: both sides, priority MOC by stamp, then limit by price then stamp */
  for (int side = -1; side <= 1; side += 2) {
    int rem = bestV;
    /* collect executable indices */
    int idx[MAXO + 16], n = 0;
    for (int i = 0; i < s->no; i++) if (side > 0 ? buy_exec(&s->o[i], x) : sell_exec(&s->o[i], x)) idx[n++] = i;
    /* sort by priority */
    for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) {
      Order *p = &s->o[idx[a]], *q = &s->o[idx[b]];
      int pp = p->type == MOC ? 0 : 1, qq = q->type == MOC ? 0 : 1;
      int swap = 0;
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
  s->no = base; /* drop continuous book orders */
  return x;
}

/* ---------------- tables ---------------- */
typedef struct { uint64_t key; float r[NA]; float sm[NA]; float bl[NA]; } Entry;
static double BETA = 0.05; static int USE_VR = 1;
typedef struct { Entry *e; uint64_t mask; long n, cap, overflow; } Table;
static void tinit(Table *t, int logsz) { t->mask = (1ULL << logsz) - 1; t->e = calloc(t->mask + 1, sizeof(Entry)); t->n = 0; t->cap = (long)((t->mask + 1) * 0.7); t->overflow = 0; }
static void tfree(Table *t) { free(t->e); }
static Entry *tget(Table *t, uint64_t key, int create) {
  if (key == 0) key = 1;
  uint64_t i = key & t->mask;
  while (1) { Entry *e = &t->e[i]; if (e->key == key) return e; if (e->key == 0) { if (!create) return NULL;
      if (t->n >= t->cap) { t->overflow++; return NULL; } e->key = key; t->n++; return e; } i = (i + 1) & t->mask; }
}

static void rm_policy(Entry *e, int mask, double *p) {
  double z = 0; int nl = 0;
  for (int a = 0; a < NA; a++) if (mask >> a & 1) { nl++; if (e && e->r[a] > 0) z += e->r[a]; }
  for (int a = 0; a < NA; a++) p[a] = (mask >> a & 1) ? (z > 0 ? (e->r[a] > 0 ? e->r[a] / z : 0) : 1.0 / nl) : 0;
}
static int avg_policy(Entry *e, int mask, double *p) { /* returns 1 if known */
  double z = 0; int nl = 0;
  for (int a = 0; a < NA; a++) if (mask >> a & 1) { nl++; if (e) z += e->sm[a]; }
  for (int a = 0; a < NA; a++) p[a] = (mask >> a & 1) ? (z > 0 ? e->sm[a] / z : 1.0 / nl) : 0;
  return z > 0;
}

/* ---------------- keys ---------------- */
/* variants: 0 min, 1 coarse, 2 coarse+trend, 3 phiH perfect recall (decision points), 4 full PR (decision points),
   5 full PR with all 330 publications */
static uint64_t agent_key(State *s, int var, int pre, int k) {
  uint64_t h = mix(1, pre); h = mix(h, k);
  /* own private state: resting stamps + post commitments */
  for (int i = 0; i < s->a_rest; i++) h = mix(h, 100 + s->a_stamps[i]);
  h = mix(h, s->a_post[0] * 16 + s->a_post[1] * 4 + s->a_post[2]);
  if (var == 0) return h;
  int nUb = s->nU;
  h = mix(h, 50 + nUb);
  if (!pre) h = mix(h, 200 + s->imbb);
  if (var == 1) return h;
  if (!pre) { h = mix(h, 300 + s->a_previmb); int mm = s->m; if (mm > 2) mm = 2; if (mm < -2) mm = -2; h = mix(h, 400 + mm); }
  if (var == 2) return h;
  /* perfect recall of signals and of observations at own decision points */
  for (int i = 0; i < s->nsig; i++) h = mix(h, 500 + s->sig[i]);
  h = mix(h, s->a_obs); h = mix(h, s->a_postacts);
  if (var == 3) return h;
  h = mix(h, s->a_prehist);
  if (var == 4) return h;
  h = mix(h, s->a_allobs);
  return h;
}
static uint64_t lp_key(State *s, int var, int k) {
  uint64_t h = mix(7, k); h = mix(h, s->l_sell * 8 + s->l_buy);
  if (var == 0) return h;
  h = mix(h, 50 + s->nU); h = mix(h, 200 + s->imbb);
  if (var == 1) return h;
  h = mix(h, 300 + s->l_previmb); int mm = s->m; if (mm > 2) mm = 2; if (mm < -2) mm = -2; h = mix(h, 400 + mm);
  if (var == 2) return h;
  for (int i = 0; i < s->nsig; i++) h = mix(h, 500 + s->sig[i]);
  h = mix(h, s->l_obs); h = mix(h, s->l_acts);
  if (var <= 4) return h;
  h = mix(h, s->a_allobs);
  return h;
}

/* ---------------- policies ---------------- */
typedef struct {
  int kind;         /* 0 table (cfr current / avg), 1 heuristic */
  Table *t; int var; int use_avg; int heur;
} Pol;

/* heuristic agent policies:
   1 MOC early: submit all at pre-cutoff period 0
   2 MOC late-dark: submit all at last pre-cutoff period
   3 imbalance-aware: wait the dark window; at decision before 15:55 submit MOC if imbalance <= 0 else
     LOC at +1 at the last decision before 15:58; IO at the very end for the rest
   heuristic LP: 1 sell IO at the last decision if imbalance >= 2, buy IO if <= -2 */
static int heur_agent(int h, State *s, int pre, int k, int mask) {
  if (h == 1) { if (pre && (mask & 2)) return 1; return 0; }
  if (h == 2) { if (pre && k >= C - Q && (mask & 2)) return 1; return 0; }
  if (h == 3) {
    if (pre) return 0;
    int e = decEpoch[k];
    if (e == 29 && s->imb <= 0 && (mask & 2)) return 1;
    if (e == 209 && (mask & 4)) return 2;
    if (e == NE - 1 && (mask & 8)) return 3;
    return 0;
  }
  return 0;
}
static int heur_lp(int h, State *s, int k, int mask) {
  if (h == 1) { if (decEpoch[k] == NE - 1) { if (s->imb >= 2 && (mask & 2)) return 1; if (s->imb <= -2 && (mask & 4)) return 2; } return 0; }
  return 0;
}

/* ---------------- episode ---------------- */
typedef struct { int player; Entry *e; int mask; double pol[NA], spol[NA]; int a; double oppr, sampr, myr; } Node;
static Node nodes[2 * (MAXC + NE) + 8]; static int nn;

typedef struct {
  int train;        /* 0 eval, 1 train */
  int trav;         /* traverser player (1/2), 0 none */
  int learn[3];     /* which players update tables */
  double eps; double iterw;
  Pol pol[3];
  long unseen[3], decisions[3];
} Ctx;

static double g_reach[3], g_samp; static int g_adv = 0;
static int choose(Ctx *c, State *s, int player, int pre, int k, int mask) {
  Pol *P = &c->pol[player];
  double pol[NA], spol[NA];
  Entry *e = NULL; int a;
  c->decisions[player]++;
  if (P->kind == 1) {
    a = player == 1 ? heur_agent(P->heur, s, pre, k, mask) : heur_lp(P->heur, s, k, mask);
    if (!(mask >> a & 1)) a = 0;
    for (int b = 0; b < NA; b++) pol[b] = b == a; memcpy(spol, pol, sizeof pol);
  } else {
    uint64_t key = player == 1 ? agent_key(s, P->var, pre, k) : lp_key(s, P->var, k);
    int learning = c->train && c->learn[player];
    e = tget(P->t, key, learning);
    if (learning) rm_policy(e, mask, pol);
    else if (P->use_avg) { if (!avg_policy(e, mask, pol)) c->unseen[player]++; }
    else { if (!e) c->unseen[player]++; rm_policy(e, mask, pol); }
    if (c->train && c->trav == player) { int nl = __builtin_popcount(mask);
      for (int b = 0; b < NA; b++) spol[b] = (mask >> b & 1) ? c->eps / nl + (1 - c->eps) * pol[b] : 0; }
    else memcpy(spol, pol, sizeof pol);
    double u = U01(), acc = 0; a = -1;
    for (int b = 0; b < NA; b++) if (mask >> b & 1) { acc += spol[b]; if (u < acc) { a = b; break; } }
    if (a < 0) for (int b = NA - 1; b >= 0; b--) if (mask >> b & 1) { a = b; break; }
  }
  if (c->train && P->kind == 0 && c->learn[player]) {
    Node *nd = &nodes[nn++]; nd->player = player; nd->e = e; nd->mask = mask; memcpy(nd->pol, pol, sizeof pol); memcpy(nd->spol, spol, sizeof spol);
    nd->a = a; nd->oppr = g_reach[3 - player]; nd->sampr = g_samp; nd->myr = g_reach[player];
    /* average strategy at non-traverser learning nodes (stochastically weighted) */
    if (c->trav != player && e) { double w = c->iterw * g_reach[player] / g_samp; for (int b = 0; b < NA; b++) e->sm[b] += (float)(w * pol[b]); }
  }
  g_reach[player] *= pol[a]; g_samp *= spol[a];
  return a;
}

/* plays one episode; returns utilities in u[1], u[2] */
static void episode(Ctx *c, double *u, int *outq) {
  static State S; State *s = &S; memset(s, 0, sizeof *s);
  nn = 0;
  g_reach[0] = g_reach[1] = g_reach[2] = 1; g_samp = 1;
  /* chance: theta */
  double u0 = U01(); s->theta = u0 < 0.35 ? 0 : (u0 < 0.65 ? 1 : 2);
  double pbuy[3] = {0.45, 0.25, 0.10}, psell[3] = {0.10, 0.25, 0.45}, pU[3] = {0.65, 0.5, 0.35};
  s->a_prehist = 11; s->a_obs = 13; s->a_postacts = 17; s->a_allobs = 19; s->l_obs = 23; s->l_acts = 29;
  s->a_previmb = 3; s->l_previmb = 3;
  /* ---- dark window ---- */
  for (int t = 0; t < C; t++) {
    int sg = U01() < pU[s->theta] ? 1 : 0; s->sig[s->nsig++] = sg; s->nU += sg;
    int mask = 1; if (committed_agent(s) < Q) mask |= 2; if (s->a_rest > 0) mask |= 4;
    int a = choose(c, s, 1, 1, t, mask);
    if (a == 1) { add_order(s, +1, MOC, 1, 99); s->a_stamps[s->a_rest++] = t; }
    else if (a == 2) { /* cancel newest agent MOC */
      for (int i = s->no - 1; i >= 0; i--) if (s->o[i].owner == 1 && s->o[i].active && s->o[i].type == MOC) { s->o[i].active = 0; break; }
      s->a_rest--; }
    s->a_prehist = mix(s->a_prehist, t * 4 + a);
    /* exogenous index MOC */
    if (U01() < pbuy[s->theta]) add_order(s, +1, MOC, 0, 99);
    if (U01() < psell[s->theta]) add_order(s, -1, MOC, 0, -99);
  }
  /* ---- 15:50: NOII publications, no cancellation ---- */
  s->m = 0;
  for (int e = 0; e < NE; e++) {
    int ts = tsec(e);
    /* exogenous events since previous publication */
    double dt = e < 30 ? 10.0 : 1.0;
    int imb0 = imbalance(s);
    if (U01() < 0.004 * dt) { int dir = U01() < 0.5 + 0.15 * (imb0 > 0) - 0.15 * (imb0 < 0) ? 1 : -1; s->m += dir; }
    if (ts < 300) { if (U01() < 0.004 * dt * (s->theta == 0 ? 2 : 1)) add_order(s, +1, MOC, 0, 99);
                    if (U01() < 0.004 * dt * (s->theta == 2 ? 2 : 1)) add_order(s, -1, MOC, 0, -99); }
    if (imb0 != 0 && U01() < ALPHA * dt * abs(imb0)) add_order(s, imb0 > 0 ? -1 : +1, IO, 0, s->m);
    s->imb = imbalance(s); s->imbb = imb_bucket(s->imb);
    uint64_t ob = s->imbb * 16 + (s->m + 8);
    s->a_allobs = mix(s->a_allobs, ob);
    if (isDec[e]) {
      int k = isDec[e] - 1;
      /* agent */
      int mask = 1;
      if (committed_agent(s) < Q) { if (ts < 300) mask |= 2; if (ts < 480) mask |= 4; mask |= 8; }
      s->a_obs = mix(s->a_obs, ob);
      int a = choose(c, s, 1, 0, k, mask);
      if (a == 1) { add_order(s, +1, MOC, 1, 99); s->a_post[0]++; }
      if (a == 2) { add_order(s, +1, LOC, 1, s->m + 1); s->a_post[1]++; }
      if (a == 3) { add_order(s, +1, IO, 1, s->m); s->a_post[2]++; }
      s->a_postacts = mix(s->a_postacts, k * 4 + a); s->a_previmb = s->imbb;
      /* LP (does not see the agent's action of this epoch) */
      int lm = 1; if (s->l_sell < LPMAX) lm |= 2; if (s->l_buy < LPMAX) lm |= 4;
      s->l_obs = mix(s->l_obs, ob);
      int b = choose(c, s, 2, 0, k, lm);
      if (b == 1) { add_order(s, -1, IO, 2, s->m + 1); s->l_sell++; }
      if (b == 2) { add_order(s, +1, IO, 2, s->m - 1); s->l_buy++; }
      s->l_acts = mix(s->l_acts, k * 4 + b); s->l_previmb = s->imbb;
    }
  }
  /* ---- cross ---- */
  int fills[MAXO + 16]; int v = s->m;
  int p = cross(s, fills);
  int q1 = 0, sold = 0, bought = 0;
  for (int i = 0; i < s->no; i++) if (fills[i]) { if (s->o[i].owner == 1) q1++; if (s->o[i].owner == 2) { if (s->o[i].side < 0) sold++; else bought++; } }
  u[1] = -(double)p * q1 - LAM * (Q - q1);
  u[2] = (double)sold * (p - v) + (double)bought * (v - p);
  if (outq) { outq[0] = q1; outq[1] = p; outq[2] = sold + bought; }
}

/* backward pass: OS-MCCFR regret updates for the traverser, with VR-MCCFR action baselines per key
   (Schmid et al. 2019): u_hat(b) = bl(b) + 1[b=a] (value - bl(b)) / spol(a); unbiased for any baseline. */
static void update(Ctx *c, double util) {
  double value = util;
  for (int i = nn - 1; i >= 0; i--) {
    Node *nd = &nodes[i];
    if (nd->player != c->trav) continue; /* opponent sampled on-policy: value unchanged */
    if (!nd->e) { value = value * nd->pol[nd->a] / nd->spol[nd->a]; continue; }
    Entry *e = nd->e; double uh[NA], vest = 0;
    for (int b = 0; b < NA; b++) if (nd->mask >> b & 1) {
      double bl = USE_VR ? e->bl[b] : 0;
      uh[b] = bl + (b == nd->a ? (value - bl) / nd->spol[nd->a] : 0);
      vest += nd->pol[b] * uh[b];
    }
    double w = nd->oppr / nd->sampr;
    for (int b = 0; b < NA; b++) if (nd->mask >> b & 1) {
      float nr = e->r[b] + (float)(w * (uh[b] - vest)); e->r[b] = nr > 0 ? nr : 0; /* RM+ */
    }
    if (USE_VR) e->bl[nd->a] += (float)(BETA * (value - e->bl[nd->a]));
    value = vest;
  }
}

/* ---------------- drivers ---------------- */
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static void train(Ctx *c, long iters, int players_mask) {
  c->train = 1;
  for (long it = 1; it <= iters; it++) {
    int trav = (players_mask == 3) ? (int)(it & 1) + 1 : players_mask;
    c->trav = trav; c->iterw = (double)it / iters; /* linear averaging */
    double u[3]; episode(c, u, NULL); if (g_adv && trav == 2) u[2] = -u[1]; update(c, u[trav]);
  }
  c->train = 0; c->trav = 0;
}

typedef struct { double m1, s1, m2, s2, fill, price; } EvalRes;
static EvalRes evaluate(Ctx *c, long n) {
  c->train = 0; c->trav = 0; c->unseen[1] = c->unseen[2] = c->decisions[1] = c->decisions[2] = 0;
  double a1 = 0, b1 = 0, a2 = 0, b2 = 0, f = 0, pr = 0;
  for (long i = 0; i < n; i++) { double u[3]; int q[3]; episode(c, u, q); a1 += u[1]; b1 += u[1] * u[1]; a2 += u[2]; b2 += u[2] * u[2]; f += q[0]; pr += q[1]; }
  EvalRes r; r.m1 = a1 / n; r.s1 = sqrt((b1 / n - r.m1 * r.m1) / n); r.m2 = a2 / n; r.s2 = sqrt((b2 / n - r.m2 * r.m2) / n);
  r.fill = f / n / Q; r.price = pr / n; return r;
}

/* best response of `player` (1 agent, 2 LP) against the fixed other player of `base`;
   adv=1: LP maximizes -u1. Returns evaluation of the resulting profile. Takes the better of key variants 0,1. */
static EvalRes br_eval(Ctx *base, int player, int adv, long brit, long neval) {
  EvalRes best; int have = 0;
  for (int v = 0; v <= 1; v++) {
    Table T; tinit(&T, 22);
    Ctx cb = *base; cb.pol[player] = (Pol){0, &T, v, 0, 0}; cb.learn[1] = cb.learn[2] = 0; cb.learn[player] = 1;
    g_adv = adv; train(&cb, brit, player); g_adv = 0;
    EvalRes r = evaluate(&cb, neval);
    double score = player == 1 ? r.m1 : (adv ? -r.m1 : r.m2);
    double bscore = !have ? -1e9 : (player == 1 ? best.m1 : (adv ? -best.m1 : best.m2));
    if (score > bscore) { best = r; have = 1; }
    tfree(&T);
  }
  return best;
}

int main(int argc, char **argv) {
  if (getenv("NOVR")) USE_VR = 0;
  if (getenv("LAM")) LAM = atof(getenv("LAM"));
  /* modes:
     train VAR_A VAR_L ITERS D SEED BRITERS NEVAL -> self-play CFR, eval, best responses
     heur HA D SEED BRITERS NEVAL                -> heuristic agent vs heuristic LP and best responses
     speed D                                      -> throughput and lookup latency */
  if (argc < 2) { fprintf(stderr, "usage\n"); return 1; }
  const char *mode = argv[1];
  if (!strcmp(mode, "train") || !strcmp(mode, "heur")) {
    int istrain = !strcmp(mode, "train");
    int va = 0, vl = 0, ha = 0, d; long iters = 0, brit, neval; uint64_t seed;
    if (istrain) { va = atoi(argv[2]); vl = atoi(argv[3]); iters = atol(argv[4]); d = atoi(argv[5]); seed = atoll(argv[6]); brit = atol(argv[7]); neval = atol(argv[8]); }
    else { ha = atoi(argv[2]); d = atoi(argv[3]); seed = atoll(argv[4]); brit = atol(argv[5]); neval = atol(argv[6]); }
    rseed(seed); make_schedule(d);
    Table TA, TL; tinit(&TA, 22); tinit(&TL, 22);
    Ctx c; memset(&c, 0, sizeof c); c.eps = 0.6;
    double tt = 0;
    if (istrain) {
      c.pol[1] = (Pol){0, &TA, va, 1, 0}; c.pol[2] = (Pol){0, &TL, vl, 1, 0}; c.learn[1] = c.learn[2] = 1;
      double t0 = now(); train(&c, iters, 3); tt = now() - t0;
    } else { c.pol[1] = (Pol){1, NULL, 0, 0, ha}; c.pol[2] = (Pol){1, NULL, 0, 0, 1}; }
    c.learn[1] = c.learn[2] = 0;
    EvalRes r = evaluate(&c, neval);
    double unseenA = c.decisions[1] ? (double)c.unseen[1] / c.decisions[1] : 0, unseenL = c.decisions[2] ? (double)c.unseen[2] / c.decisions[2] : 0;
    Ctx ch = c; ch.pol[2] = (Pol){1, NULL, 0, 0, 1}; EvalRes rh = evaluate(&ch, neval);   /* heuristic LP */
    Ctx cw = c; cw.pol[2] = (Pol){1, NULL, 0, 0, 0}; EvalRes rw = evaluate(&cw, neval);   /* LP withholds liquidity */
    double xh[4] = {0, 0, 0, 0};
    if (istrain) for (int h = 1; h <= 3; h += 2) { Ctx cx = c; cx.pol[1] = (Pol){1, NULL, 0, 0, h}; xh[h] = evaluate(&cx, neval).m1; }
    EvalRes rb = br_eval(&c, 2, 0, brit, neval);   /* LP best response (own profit) */
    EvalRes rv = br_eval(&c, 2, 1, brit, neval);   /* adversarial LP */
    EvalRes ra = br_eval(&c, 1, 0, brit, neval);   /* agent best response to the LP */
    double worst = rw.m1 < rv.m1 ? rw.m1 : rv.m1; if (rh.m1 < worst) worst = rh.m1;
    printf("{\"lam\":%.2f,\"mode\":\"%s\",\"var_agent\":%d,\"var_lp\":%d,\"heur_agent\":%d,\"iters\":%ld,\"D\":%d,\"seed\":%llu,\"train_sec\":%.2f,"
           "\"entries_agent\":%ld,\"entries_lp\":%ld,\"overflow\":%ld,"
           "\"u1\":%.5f,\"u1_se\":%.5f,\"u2\":%.5f,\"fill\":%.4f,\"price\":%.4f,\"unseen_agent\":%.4f,\"unseen_lp\":%.4f,"
           "\"u1_vs_heurLP\":%.5f,\"u1_vs_withhold\":%.5f,\"u1_vs_advLP\":%.5f,\"u1_worst\":%.5f,"
           "\"u2_brLP\":%.5f,\"u1_brA\":%.5f,\"gain_lp\":%.5f,\"gain_agent\":%.5f,\"nashconv_lb\":%.5f,\"u1_heur1_vs_LP\":%.5f,\"u1_heur3_vs_LP\":%.5f}\n",
           LAM, mode, va, vl, ha, iters, D, (unsigned long long)seed, tt, TA.n, TL.n, TA.overflow + TL.overflow,
           r.m1, r.s1, r.m2, r.fill, r.price, unseenA, unseenL,
           rh.m1, rw.m1, rv.m1, worst, rb.m2, ra.m1, rb.m2 - r.m2, ra.m1 - r.m1, (rb.m2 - r.m2) + (ra.m1 - r.m1), xh[1], xh[3]);
    return 0;
  }
  if (!strcmp(mode, "speed")) {
    int d = atoi(argv[2]); rseed(1); make_schedule(d);
    Table TA, TL; tinit(&TA, 22); tinit(&TL, 22);
    Ctx c; memset(&c, 0, sizeof c); c.eps = 0.6; c.pol[1] = (Pol){0, &TA, 2, 0, 0}; c.pol[2] = (Pol){0, &TL, 2, 0, 0}; c.learn[1] = c.learn[2] = 1;
    double t0 = now(); train(&c, 200000, 3); double t1 = now();
    /* decision latency: table lookup + policy */
    State s; memset(&s, 0, sizeof s); double p[NA]; volatile double acc = 0;
    double t2 = now(); for (int i = 0; i < 1000000; i++) { s.imbb = i % 7; Entry *e = tget(&TA, agent_key(&s, 2, 0, i % D), 0); avg_policy(e, 15, p); acc += p[0]; } double t3 = now();
    printf("{\"D\":%d,\"us_per_iteration\":%.3f,\"ns_per_decision_lookup\":%.1f}\n", D, (t1 - t0) / 200000 * 1e6, (t3 - t2) / 1e6 * 1e9);
    return 0;
  }
  return 1;
}
