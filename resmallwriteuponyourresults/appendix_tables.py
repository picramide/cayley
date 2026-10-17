"""Appendix tables: baselines, compute, graph statistics across lengths; random-circulant figure."""
import json, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from masks import *
from graph_analysis import receptive_field, hops_to_cover, n_components, adjacent_coverage, spectral_gap
rows = [json.loads(l) for l in open(sys.argv[1])]
R = {(r['task_name'], r['mask_name']): r for r in rows if r['num_train_epochs'] == 5.0}
TASKS = ['cola','sst2','mrpc','qqp','stsb','rte','qnli','mnli']
TL = {'cola':'CoLA','sst2':'SST-2','mrpc':'MRPC','qqp':'QQP','stsb':'STS-B','rte':'RTE','qnli':'QNLI','mnli':'MNLI'}
MASKS = ['dense','local','window_dilations','circulant','hypercube','random_circulant','bigbird']
LAB = {'dense':'Dense','local':'Local ($r{=}16$)','window_dilations':'Window + dil.','circulant':'Circulant $\\{\\pm1,\\pm2,\\pm3\\}$',
       'hypercube':'Hypercube $Q_7$','random_circulant':'Random circulant$^\\ast$','bigbird':'BigBird-style$^\\ast$','hypercube_gray':'Hypercube, Gray (Wang et al.)'}

# --- dev-set class counts, recovered from degenerate runs (accuracy x size) --------------
def size(t, m='dense', pre='eval_'):
    met = R[(t, m)]['metrics']
    return sum(v for k, v in met.items() if k.startswith(pre + 'predicted_class') or k.startswith('eval_matched_predicted_class' if pre=='eval_matched_' else 'zz'))
N_DEV = {'cola':1043,'sst2':872,'mrpc':408,'qqp':40430,'rte':277,'qnli':5463,'mnli':9815,'stsb':1500}
counts = {
 'cola': 'acceptable 721, unacceptable 322$^{a}$',
 'sst2': 'positive 444, negative 428$^{a}$',
 'mrpc': f"paraphrase {round(R[('mrpc','circulant')]['metrics']['eval_accuracy']*408)}, not {408-round(R[('mrpc','circulant')]['metrics']['eval_accuracy']*408)}",
 'qqp':  f"non-duplicate {round(R[('qqp','circulant')]['metrics']['eval_accuracy']*40430)}, duplicate {40430-round(R[('qqp','circulant')]['metrics']['eval_accuracy']*40430)}",
 'stsb': 'regression, 1{,}500 pairs',
 'rte':  f"class 0: {round(R[('rte','hypercube')]['metrics']['eval_accuracy']*277)}, class 1: {277-round(R[('rte','hypercube')]['metrics']['eval_accuracy']*277)}",
 'qnli': f"class 1: {round(R[('qnli','circulant')]['metrics']['eval_accuracy']*5463)}, class 0: {5463-round(R[('qnli','circulant')]['metrics']['eval_accuracy']*5463)}",
 'mnli': None,
}
m5 = R[('mnli','local')]['metrics']; import json as _j
c0 = round(m5['eval_matched_accuracy']*9815)
r3 = [json.loads(l) for l in open(sys.argv[1])]
m3 = [r for r in r3 if r['task_name']=='mnli' and r['mask_name']=='local' and r['num_train_epochs']==3.0][0]['metrics']
c1 = round(m3['eval_matched_accuracy']*9815)
counts['mnli'] = f"matched: {c0} / {c1} / {9815-c0-c1}"
base = {'cola':('MCC','constant $\\Rightarrow$ 0 (undefined; reported as 0)','0.000'),
        'sst2':('Accuracy','always positive','50.92'),
        'mrpc':('F1 (positive)','always paraphrase: $\\tfrac{2\\cdot279}{2\\cdot279+129}$','81.22'),
        'qqp':('F1 (duplicate)','always duplicate: $\\tfrac{2p}{1+p}$, $p=\\tfrac{14885}{40430}$','53.82'),
        'stsb':('Pearson','constant $\\Rightarrow$ undefined; reported as 0','0.000'),
        'rte':('Accuracy','always class 0','52.71'),
        'qnli':('Accuracy','always class 1','50.54'),
        'mnli':('Accuracy (m)','always class 0','35.45')}
with open('tables/baselines.tex','w') as f:
    for t in TASKS:
        f.write(f"{TL[t]} & {N_DEV[t]:,} & {counts[t]} & {base[t][0]} & {base[t][1]} & {base[t][2]} \\\\\n".replace(',',"{,}",1) if False else
                f"{TL[t]} & {N_DEV[t]:,} & {counts[t]} & {base[t][0]} & {base[t][1]} & {base[t][2]} \\\\\n")
    f.write('\\bottomrule\n')

# --- compute ------------------------------------------------------------------------------
tot_h = sum(r['train_metrics']['train_runtime'] for r in rows) / 3600
with open('tables/compute.tex','w') as f:
    for t in TASKS:
        d = R[(t,'dense')]['train_metrics']
        rat = [R[(t,m)]['train_metrics']['train_runtime']/d['train_runtime'] for m in MASKS[1:]]
        fl = {R[(t,m)]['train_metrics']['total_flos'] for m in MASKS}
        f.write(f"{TL[t]} & {d['train_runtime']/60:.1f} & {min(rat):.2f}--{max(rat):.2f} & {d['train_samples_per_second']:.0f} & {list(fl)[0]:.2e} & {'yes' if len(fl)==1 else 'no'} \\\\\n")
    f.write('\\bottomrule\n')
print('total train hours (all 63 runs): %.1f' % tot_h)

# --- graph statistics across lengths -----------------------------------------------------
Ls = [8, 16, 24, 32, 48, 64, 96, 112, 128]
M = dict(STUDIED); M['hypercube_gray'] = hypercube(labeling='gray')
order = MASKS + ['hypercube_gray']
def cell_h(A, L):
    h = hops_to_cover(A, L); k = n_components(A, L)
    return '$\\infty$ (%d)' % k if not np.isfinite(h) else str(int(h))
with open('tables/hops_by_L.tex','w') as f:
    for m in order:
        f.write(LAB[m] + ' & ' + ' & '.join(cell_h(M[m], L) for L in Ls) + ' \\\\\n')
    f.write('\\bottomrule\n')
with open('tables/adj_by_L.tex','w') as f:
    for m in order:
        f.write(LAB[m] + ' & ' + ' & '.join(f'{adjacent_coverage(M[m], L):.2f}' for L in Ls) + ' \\\\\n')
    f.write('\\bottomrule\n')

# --- random circulant family -------------------------------------------------------------
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
                     'axes.linewidth': 0.6, 'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
BLUE, ORANGE, AQUA, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#8a8984'
draws = [random_circulant(seed=s) for s in range(200)]
Lr = np.arange(2, 129)
pdisc = [np.mean([n_components(A, L) > 1 for A, _ in draws]) for L in Lr]
pnot12 = [np.mean([hops_to_cover(A, L) > 12 for A, _ in draws]) for L in Lr]
gaps = [spectral_gap(A) for A, _ in draws]
adj1 = np.mean([(1 in g) or (127 in g) for _, g in draws])
fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.1))
a.plot(Lr, pdisc, color=BLUE, lw=1.6, label='$G_L$ disconnected')
a.plot(Lr, pnot12, color=ORANGE, lw=1.6, ls='--', label='position 0 cannot see all\nvalid tokens in 12 layers')
a.set_xlabel('valid tokens $L$'); a.set_ylabel('fraction of 200 draws'); a.set_ylim(0, 1.02)
a.legend(frameon=False, fontsize=6.5); a.set_title('(a) padding and reach', fontsize=8, loc='left')
b.hist(gaps, bins=24, color=BLUE, edgecolor='white', lw=0.6)
for name, A, ls in [('hypercube', hypercube(), '--'), ('window+dil.', window_dilations(), ':'), ('local', local(16), '-.')]:
    x = spectral_gap(A); b.axvline(x, color=INK, lw=1, ls=ls); b.text(x + 0.005, b.get_ylim()[1]*0.9 if name!='local' else b.get_ylim()[1]*0.6, name, fontsize=6.5, rotation=90, va='top')
b.set_xlabel('spectral gap $\\lambda_2$ (full mask)'); b.set_ylabel('draws'); b.set_title('(b) expansion of random draws', fontsize=8, loc='left')
plt.tight_layout(pad=0.4); plt.savefig('figures/random_circulant_family.pdf', bbox_inches='tight'); plt.close()
stats = dict(adj_frac=float(adj1), gap_median=float(np.median(gaps)), gap_min=float(min(gaps)), gap_max=float(max(gaps)),
             frac_gap_above_hyper=float(np.mean(np.array(gaps) > spectral_gap(hypercube()))),
             pdisc16=float(pdisc[list(Lr).index(16)]), pdisc32=float(pdisc[list(Lr).index(32)]), pdisc64=float(pdisc[list(Lr).index(64)]),
             pnot12_16=float(pnot12[list(Lr).index(16)]), pnot12_32=float(pnot12[list(Lr).index(32)]), pnot12_64=float(pnot12[list(Lr).index(64)]))
json.dump(stats, open('data/random_circulant_family.json', 'w'), indent=1); print(stats)
