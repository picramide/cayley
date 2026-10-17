import json, sys
rows=[json.loads(l) for l in open(sys.argv[1])]
R={(r['task_name'],r['mask_name']):r for r in rows if r['num_train_epochs']==5.0}
R3={(r['task_name'],r['mask_name']):r for r in rows if r['num_train_epochs']==3.0}
c=json.load(open('data/canonical.json')); g=json.load(open('data/graph_stats.json'))
MASKS=['dense','local','window_dilations','circulant','hypercube','random_circulant','bigbird']
LAB={'dense':'Dense','local':'Local ($r{=}16$)','window_dilations':'Window + dilations','circulant':'Circulant $\\{\\pm1,\\pm2,\\pm3\\}$',
     'hypercube':'Hypercube $Q_7$','random_circulant':'Random circulant','bigbird':'BigBird-style'}
DEG={'dense':128,'local':'$\\le$33','window_dilations':36,'circulant':7,'hypercube':8,'random_circulant':17,'bigbird':'$\\approx$16'}
def deg(m,r):
    if m=='mnli': pass
def D(t,m): return c[f'{t}|{m}']['degenerate']
def f(x,pct=True,dag=False):
    s=f'{100*x:.2f}' if pct else f'{x:.3f}'
    return s+('$^\\dagger$' if dag else '')
lines=[]
lines.append(r'Best constant & -- & 0.000 & 50.92 & 68.38 / 81.22 & 63.18 / 53.82 & -- & 52.71 & 50.54 & 35.45 / 35.22 \\ \midrule')
for m in MASKS:
    M=lambda t: R[(t,m)]['metrics']
    cells=[LAB[m], str(DEG[m]),
      f(M('cola')['eval_matthews_correlation'],False,D('cola',m)),
      f(M('sst2')['eval_accuracy'],dag=D('sst2',m)),
      f(M('mrpc')['eval_accuracy'])+' / '+f(M('mrpc')['eval_f1'],dag=D('mrpc',m)),
      f(M('qqp')['eval_accuracy'])+' / '+f(M('qqp')['eval_f1'],dag=D('qqp',m)),
      f(M('stsb')['eval_pearson'],False)+' / '+f(M('stsb')['eval_spearmanr'],False),
      f(M('rte')['eval_accuracy'],dag=D('rte',m)),
      f(M('qnli')['eval_accuracy'],dag=D('qnli',m)),
      f(M('mnli')['eval_matched_accuracy'])+' / '+f(M('mnli')['eval_mismatched_accuracy'],dag=D('mnli',m))]
    lines.append(' & '.join(cells)+r' \\'+(r' \midrule' if m=='dense' else ''))
open('tables/main_results.tex','w').write('\n'.join(lines)+'\n\\bottomrule\n')
# mask properties
i32=g['dense']['Ls'].index(32); i64=g['dense']['Ls'].index(64); i128=g['dense']['Ls'].index(128)
GRP={'dense':'$\\mathbb Z_{128}$','local':'(band; not Cayley)','window_dilations':'$\\mathbb Z_{128}$','circulant':'$\\mathbb Z_{128}$',
     'hypercube':'$\\mathbb Z_2^{7}$','random_circulant':'$\\mathbb Z_{128}$','bigbird':'(not Cayley)'}
pl=[]
for m in MASKS:
    d=g[m]; star='$^\\ast$' if m in ('random_circulant','bigbird') else ''
    adj=d['adj_cov'][i32]
    if m=='random_circulant': adjs='0.00'   # exact for the run's mask, see text
    else: adjs=f'{adj:.2f}'
    pl.append(' & '.join([LAB[m],GRP[m],f"{d['degree']:.1f}",f"{d['density']:.3f}",f"{d['spectral_gap_full']:.3f}{star}",adjs,
                          f"{d['reach'][i64]:.2f}{star}", f"{d['hops'][i128]:.0f}{star}"])+r' \\')
open('tables/mask_props.tex','w').write('\n'.join(pl)+'\n\\bottomrule\n')
# appendix: 3 vs 5 epoch MNLI, train loss, best epoch
al=[]
for m in MASKS:
    a=R3[('mnli',m)]; b=R[('mnli',m)]
    al.append(' & '.join([LAB[m], f(a['metrics']['eval_matched_accuracy'],dag=c['mnli|'+m]['degenerate'] or m not in('dense','bigbird')), f"{a['train_metrics']['train_loss']:.4f}",
                          f(b['metrics']['eval_matched_accuracy'],dag=D('mnli',m)), f"{b['train_metrics']['train_loss']:.4f}"])+r' \\')
open('tables/mnli_epochs.tex','w').write('\n'.join(al)+'\n\\bottomrule\n')
# appendix: train loss + best epoch for all tasks
TASKS=['cola','sst2','mrpc','qqp','stsb','rte','qnli','mnli']
tl=[]
for m in MASKS:
    tl.append(' & '.join([LAB[m]]+[f"{R[(t,m)]['train_metrics']['train_loss']:.3f} ({int(R[(t,m)]['best_epoch'])})" for t in TASKS])+r' \\')
open('tables/train_loss.tex','w').write('\n'.join(tl)+'\n\\bottomrule\n')
# appendix: sha256 (task x mask grid, first 8 hex digits)
TLs={'cola':'CoLA','sst2':'SST-2','mrpc':'MRPC','qqp':'QQP','stsb':'STS-B','rte':'RTE','qnli':'QNLI','mnli':'MNLI'}
sl=[]
for t in TASKS:
    sl.append(TLs[t]+' & '+' & '.join(f"\\texttt{{{R[(t,m)]['mask_sha256'][:8]}}}" for m in MASKS[1:])+r' \\')
open('tables/sha.tex','w').write('\n'.join(sl)+'\n\\bottomrule\n')
# appendix: valid-pair density by L
import sys as _s; _s.path.insert(0,'scripts')
from masks import STUDIED, hypercube as _hc, cayley_Zn as _cz
_M=dict(STUDIED); _M['hypercube_gray']=_hc(labeling='gray'); _M['dyadic']=_cz([1,2,4,8,16,32,64])
_LAB=dict(LAB, hypercube_gray='Hypercube, Gray (Wang et al.)', dyadic='Dyadic circulant (proposed)')
_LAB['random_circulant']+='$^\\ast$'; _LAB['bigbird']+='$^\\ast$'
_Ls=[8,12,17,24,32,48,64,96,128]
vl=[]
for m in ['dense','local','window_dilations','circulant','dyadic','hypercube','hypercube_gray','random_circulant','bigbird']:
    A=_M[m]; vl.append(' & '.join([_LAB[m]]+[f"{A[:L,:L].mean():.2f}" for L in _Ls])+r' \\')
open('tables/valid_density.tex','w').write('\n'.join(vl)+'\n\\bottomrule\n')
# headroom table numeric
hl=[]
for m in MASKS[1:]:
    hl.append(' & '.join([LAB[m]]+[f"{c[f'{t}|{m}']['headroom']:.2f}"+('$^\\dagger$' if D(t,m) else '') for t in TASKS])+r' \\')
open('tables/headroom.tex','w').write('\n'.join(hl)+'\n\\bottomrule\n')
print(open('tables/main_results.tex').read()); print(open('tables/mask_props.tex').read()); print(open('tables/mnli_epochs.tex').read()); print(open('tables/train_loss.tex').read())
