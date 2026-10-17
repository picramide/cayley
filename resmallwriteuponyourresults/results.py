"""Build canonical result tables from benchmark_all.jsonl.
Canonical protocol = 5 training epochs (MNLI also has 3-epoch runs; kept as a secondary check)."""
import json, sys, os
rows=[json.loads(l) for l in open(sys.argv[1])]
out=sys.argv[2]
MASKS=['dense','local','window_dilations','circulant','hypercube','random_circulant','bigbird']
TASKS=['cola','sst2','mrpc','qqp','stsb','rte','qnli','mnli']
# majority-class / constant-predictor baselines on the GLUE dev sets, derived from
# the class counts of degenerate runs in this file (see audit) and GLUE dev label counts
# best constant predictor for the primary metric (for QQP F1 that is always-duplicate,
# not the majority class, which scores F1 = 0)
_pq=14885/40430
MAJ={'cola':0.0,'sst2':444/872,'mrpc':2*279/(2*279+129),'qqp':2*_pq/(1+_pq),'stsb':0.0,
     'rte':146/277,'qnli':2761/5463,'mnli':3479/9815}
PRIMARY={'cola':'eval_matthews_correlation','sst2':'eval_accuracy','mrpc':'eval_f1','qqp':'eval_f1',
         'stsb':'eval_pearson','rte':'eval_accuracy','qnli':'eval_accuracy','mnli':'eval_matched_accuracy'}
R={}; R3={}
for r in rows:
    k=(r['task_name'],r['mask_name'])
    (R if r['num_train_epochs']==5.0 else R3)[k]=r
def degenerate(r):
    m=r['metrics']
    cnt=[v for kk,v in m.items() if 'predicted_class' in kk and kk.startswith(('eval_predicted','eval_matched_predicted'))]
    if not cnt: return False
    return sum(1 for c in cnt if c>0)==1
def minority_frac(r):
    m=r['metrics']; cnt=[v for kk,v in m.items() if kk.startswith(('eval_predicted','eval_matched_predicted'))]
    return min(cnt)/sum(cnt) if cnt else None
tab={}
for t in TASKS:
    for mk in MASKS:
        r=R[(t,mk)]; m=r['metrics']
        s=m[PRIMARY[t]]; d=R[(t,'dense')]['metrics'][PRIMARY[t]]
        hr=(s-MAJ[t])/(d-MAJ[t])
        tab[(t,mk)]=dict(score=s,dense=d,maj=MAJ[t],headroom=hr,degenerate=degenerate(r),
                         minority=minority_frac(r),best_epoch=r['best_epoch'],metrics=m)
# sanity: MRPC majority should equal circulant (all-positive) F1
assert abs(tab[('mrpc','circulant')]['score']-MAJ['mrpc'])<1e-9
assert abs(tab[('qnli','circulant')]['score']-MAJ['qnli'])<1e-9
assert abs(tab[('mnli','local')]['score']-MAJ['mnli'])<1e-9
assert abs(tab[('rte','hypercube')]['score']-MAJ['rte'])<1e-9
json.dump({f'{t}|{m}':{k:v for k,v in d.items() if k!='metrics'} for (t,m),d in tab.items()},open(f'{out}/canonical.json','w'),indent=1)
# print summary
for mk in MASKS:
    print(f'{mk:17s}',' '.join(f"{t}:{tab[(t,mk)]['headroom']:+.2f}{'*' if tab[(t,mk)]['degenerate'] else ''}" for t in TASKS))
ndeg=sum(tab[(t,m)]['degenerate'] for t in TASKS for m in MASKS if m!='dense')
print('degenerate sparse runs (5ep):',ndeg,'of',8*6)
print('MNLI 3-ep degenerate:',[m for m in MASKS if degenerate(R3[('mnli',m)])])
print('near-degenerate (minority<5%):',[(t,m,round(tab[(t,m)]['minority'],3)) for t in TASKS for m in MASKS if tab[(t,m)]['minority'] is not None and 0<tab[(t,m)]['minority']<0.05])
print('QNLI dense vs local counts:',R[('qnli','dense')]['metrics']['eval_predicted_class_0_count'],R[('qnli','local')]['metrics']['eval_predicted_class_0_count'])
print('MNLI dense 3ep vs 5ep', R3[('mnli','dense')]['metrics']['eval_matched_accuracy'], R[('mnli','dense')]['metrics']['eval_matched_accuracy'])
print('MNLI bigbird 3ep vs 5ep', R3[('mnli','bigbird')]['metrics']['eval_matched_accuracy'], R[('mnli','bigbird')]['metrics']['eval_matched_accuracy'])
# eval runtime ratio vs dense
import statistics
for mk in MASKS:
    rat=[R[(t,mk)]['train_metrics']['train_runtime']/R[(t,'dense')]['train_metrics']['train_runtime'] for t in TASKS]
    print(mk,'train-time ratio vs dense: median %.3f'%statistics.median(rat))
