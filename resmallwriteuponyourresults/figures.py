import json, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, 'scripts'); from masks import *; from graph_analysis import receptive_field
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.linewidth': 0.6, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
INK, MUTED, BLUE, ORANGE, AQUA = '#0b0b0b', '#8a8984', '#2a78d6', '#eb6834', '#1baf7a'
c = json.load(open('data/canonical.json')); g = json.load(open('data/graph_stats.json'))
MASKS = ['dense','local','window_dilations','circulant','hypercube','random_circulant','bigbird']
LAB = {'dense':'Dense','local':'Local ($r{=}16$)','window_dilations':'Window+dil.','circulant':'Circulant $\\{1,2,3\\}$',
       'hypercube':'Hypercube $Q_7$','random_circulant':'Rand. circulant','bigbird':'BigBird-style'}
TASKS = ['cola','sst2','qqp','qnli','stsb','mnli','mrpc','rte']
TL = {'cola':'CoLA','sst2':'SST-2','mrpc':'MRPC','qqp':'QQP','stsb':'STS-B','rte':'RTE','qnli':'QNLI','mnli':'MNLI-m'}

# Fig 1: mask gallery
fig, ax = plt.subplots(1, 8, figsize=(7.0, 1.25))
gal = dict(STUDIED); gal['hypercube_gray'] = hypercube(labeling='gray')
order = MASKS + ['hypercube_gray']; lab2 = dict(LAB, hypercube_gray='Hypercube, Gray\n(Wang et al.)')
for a, m in zip(ax, order):
    a.imshow(gal[m][:64, :64], cmap=LinearSegmentedColormap.from_list('b', ['#fcfcfb', '#1c4f8f']), interpolation='nearest', vmin=0, vmax=1)
    a.set_xticks([]); a.set_yticks([]); [s.set_visible(True) for s in a.spines.values()]
    [s.set_color('#c3c2b7') for s in a.spines.values()]
    a.set_title(lab2[m] + ('$^\\ast$' if m in ('random_circulant','bigbird') else ''), fontsize=6.5, color=INK)
plt.tight_layout(pad=0.3); plt.savefig('figures/mask_gallery.pdf', bbox_inches='tight'); plt.close()

# Fig 2: headroom-recovered heatmap
H = np.array([[c[f'{t}|{m}']['headroom'] for t in TASKS] for m in MASKS[1:]])
D = np.array([[c[f'{t}|{m}']['degenerate'] for t in TASKS] for m in MASKS[1:]])
cmap = LinearSegmentedColormap.from_list('seq', ['#f4f3ef', '#a9c8f0', '#2a78d6', '#163f73'])
fig, a = plt.subplots(figsize=(5.4, 2.3))
a.imshow(np.clip(H, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect='auto')
for i in range(H.shape[0]):
    for j in range(H.shape[1]):
        v = H[i, j]; txt = f'{v:.2f}' + ('$^\\dagger$' if D[i, j] else '')
        a.text(j, i, txt, ha='center', va='center', fontsize=7, color='white' if v > 0.55 else INK)
a.set_xticks(range(len(TASKS)), [TL[t] for t in TASKS]); a.set_yticks(range(len(MASKS)-1), [LAB[m] for m in MASKS[1:]])
a.tick_params(length=0); [s.set_visible(False) for s in a.spines.values()]
a.axvline(1.5, color='white', lw=3)
a.text(0.5, -0.9, 'single sentence', ha='center', fontsize=7, color='#52514e')
a.text(4.5, -0.9, 'sentence pair', ha='center', fontsize=7, color='#52514e')
a.set_ylim(len(MASKS)-1.5, -1.2)
plt.tight_layout(pad=0.3); plt.savefig('figures/headroom.pdf', bbox_inches='tight'); plt.close()

# Fig 3: CLS receptive field vs valid length L
fig, a1 = plt.subplots(figsize=(4.6, 2.3))
Ls = np.arange(2, 129)
series = [('circulant', circulant(), BLUE, '-'), ('cwin1', circular_window(1), ORANGE, '--'),
          ('cwin2', circular_window(2), AQUA, '-.')]
names = {'circulant': 'Circulant $\\{\\pm1,\\pm2,\\pm3\\}$ (studied)', 'cwin1': 'Circular window $r{=}1$', 'cwin2': 'Circular window $r{=}2$'}
rc = np.array([[receptive_field(random_circulant(seed=s)[0], L).mean() for L in Ls] for s in range(100)])
a1.fill_between(Ls, np.percentile(rc, 10, 0), np.percentile(rc, 90, 0), color=MUTED, alpha=0.18, lw=0)
a1.plot(Ls, np.median(rc, 0), color=MUTED, lw=1.2, label='Random circulant, degree 17 (median and 10-90% band, 100 draws)')
for k, A, col, ls in series:
    y = [receptive_field(A, L).mean() for L in Ls]; a1.plot(Ls, y, color=col, ls=ls, lw=1.6, label=names[k])
a1.plot([], [], color=INK, lw=1.2, label='Dense, local, window+dil., hypercube, BigBird (all = 1)')
a1.axvline(37, color='#c3c2b7', lw=0.8, ls=':'); a1.text(39, 0.03, '$L=37$', fontsize=6.5, color='#52514e')
a1.set_xlabel('valid (non-padding) tokens $L$'); a1.set_ylabel('fraction of tokens visible to\nposition 0 after 12 layers')
a1.set_ylim(0, 1.05); a1.set_xlim(0, 129)
a1.legend(frameon=False, fontsize=6.3, loc='upper center', bbox_to_anchor=(0.5, -0.34), ncol=1)
plt.tight_layout(pad=0.4); plt.savefig('figures/reach.pdf', bbox_inches='tight'); plt.close()

# Fig 4: single-sentence headroom vs adjacency coverage and vs spectral gap
OFF = {'adj': {'local': (-6, 4), 'window_dilations': (6, 2), 'bigbird': (6, -8), 'circulant': (-6, -3)},
       'gap': {'local': (4, 4), 'window_dilations': (5, 2), 'bigbird': (-6, -10), 'circulant': (5, -9)}}
i32 = g['dense']['Ls'].index(32); sp = MASKS[1:]
fig, (b1, b2) = plt.subplots(1, 2, figsize=(6.6, 2.1), sharey=True)
for ax_, key, xl in [(b1, 'adj', 'adjacent-token coverage ($L{=}32$)'), (b2, 'gap', 'spectral gap $\\lambda_2$')]:
    for m in sp:
        y = np.mean([max(0, c[f'{t}|{m}']['headroom']) for t in ('cola', 'sst2')])
        x = g[m]['adj_cov'][i32] if key == 'adj' else g[m]['spectral_gap_full']
        ax_.scatter(x, y, s=28, color=BLUE, edgecolor='white', lw=1, zorder=3)
        off = OFF[key].get(m, (5, -3))
        ax_.annotate(LAB[m], (x, y), xytext=off, textcoords='offset points', fontsize=6.3, color=INK, ha='right' if off[0] < 0 else 'left')
    ax_.set_xlabel(xl)
b1.set_ylabel('headroom recovered\n(mean of CoLA, SST-2)'); b1.set_ylim(0, 1.08); b1.set_xlim(-0.08, 1.3); b2.set_xlim(-0.02, 0.62)
plt.tight_layout(pad=0.4); plt.savefig('figures/single_sentence_scatter.pdf', bbox_inches='tight'); plt.close()
print('ok')
