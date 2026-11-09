import json, sys
rows = [json.loads(l) for l in open('res4/scale.jsonl') if l.startswith('{')]
cols = ['var_agent','iters','train_sec','eval_sec','entries_agent','entries_lp','overflow','u1','u1_vs_heurLP','u1_vs_withhold','u1_heur1_vs_LP','unseen_agent','gain_lp','gain_agent','nashconv_lb']
print(' | '.join(c[:10] for c in cols))
for r in rows: print(' | '.join(str(round(r[c],4)) if isinstance(r[c], float) else str(r[c]) for c in cols))
