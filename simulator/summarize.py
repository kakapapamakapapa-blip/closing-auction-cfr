import json, glob, sys
d = sys.argv[1] if len(sys.argv) > 1 else 'res2'
cols = ['mode','var_agent','heur_agent','iters','D','train_sec','entries_agent','entries_lp','overflow','u1','u1_se','fill',
        'unseen_agent','u1_vs_heurLP','u1_vs_withhold','u1_vs_advLP','u1_worst','gain_lp','gain_agent','nashconv_lb','u1_heur1_vs_LP','u1_heur3_vs_LP']
for f in sorted(glob.glob(d + '/*.jsonl')):
    rows = [json.loads(l) for l in open(f) if l.strip().startswith('{')]
    if not rows: continue
    print('==', f)
    print(' | '.join(c[:9] for c in cols))
    for r in rows:
        print(' | '.join((str(round(r[c], 4)) if isinstance(r.get(c), float) else str(r.get(c, '-'))) for c in cols))
