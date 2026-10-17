import json, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
print(len(rows))
seen={}
for r in rows:
    k=(r['task_name'],r['mask_name'])
    seen.setdefault(k,[]).append(r)
for k,v in seen.items():
    if len(v)>1: print('DUP',k,[(x['num_train_epochs'],x['best_metric']) for x in v])
for r in rows:
    m=r['metrics']; pc={kk:vv for kk,vv in m.items() if 'predicted_class' in kk}
    print(r['task_name'],r['mask_name'],r['num_train_epochs'],r['best_epoch'],round(r['best_metric'],4),pc)
