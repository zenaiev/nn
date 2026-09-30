import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import uproot

np.random.seed(42)

#n = 500000
#n1 = 50000
#a = 3

#x = np.random.exponential(a, (n, 3))


data = uproot.concatenate({f: 'tree' for f in [
      'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/00000/ttbarSel_merged.root',
      'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/00001/ttbarSel_merged.root',
      'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/010000/ttbarSel_merged.root',
      'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/010001/ttbarSel_merged.root',
      'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/010002/ttbarSel_merged.root',
      'ntuples-mc/TTJets_TuneZ2_7TeV-madgraph-tauola/010003/ttbarSel_merged.root',
    ]}, ['mcT'], library='np')
x = data['mcT']
n = x.shape[0]
n1 = int(n/10)
print(f'Events: {n} -> {n1}')

x1 = x[:n1].copy()
#x2 = x[n1:].copy()

def nn(x1, x2):
  # найближчий сусід
  tree = cKDTree(x1)
  _, i = tree.query(x2)
  # ваги
  weights = np.bincount(i, minlength=n1)
  sum_x2 = np.zeros_like(x1)
  np.add.at(sum_x2, i, x2)
  #x1_corr = (x1 + sum_x2) / weights[:, None]
  x1_corr = sum_x2 / weights[:, None]
  #for obj in [x1, weights, x1_corr]: print(obj.shape)
  return x1, weights, x1_corr

# 1а ітерація
x1, weights, x1_corr = nn(x1, x)

# 2а ітерація
x1_corr, weights2, x1_corr2 = nn(x1_corr, x)

# 3а ітерація
x1_corr2, weights3, x1_corr3 = nn(x1_corr2, x)

# histograms
def obs_pt(x):
  # pT
  return np.sqrt(x[:,0]**2+x[:,1]**2)
def obs_rap(x):
  # rapidity
  e = np.sqrt(x[:,0]**2+x[:,1]**2+x[:,2]**2+x[:,3]**2)
  return 0.5 * np.log((e + x[:,2]) / (e - x[:,2]))
fig, axs = plt.subplots(
    2, 2, figsize=(15, 10),
    #sharex=True,
    #gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
)
for iobs, obs in enumerate([obs_pt, obs_rap]):
  if iobs == 0:
    bins = np.linspace(0, 300, 31)
  elif iobs == 1:
    bins = np.linspace(-2.2, 2.2, 23)
  h, _ = np.histogram(obs(x), bins)
  h1, _ = np.histogram(obs(x1), bins, weights=np.ones_like(obs(x1)) * n / n1)
  hw, _ = np.histogram(obs(x1), bins, weights=weights)
  hwcorr, _ = np.histogram(obs(x1_corr), bins, weights=weights)
  hw2, _ = np.histogram(obs(x1_corr), bins, weights=weights2)
  hwc2, _ = np.histogram(obs(x1_corr2), bins, weights=weights2)
  hw3, _ = np.histogram(obs(x1_corr2), bins, weights=weights3)
  hwc3, _ = np.histogram(obs(x1_corr3), bins, weights=weights3)
  rms1 = np.sqrt(np.mean((h1 / h - 1)**2))
  rmsw = np.sqrt(np.mean((hw / h - 1)**2))
  rmswcor = np.sqrt(np.mean((hwcorr / h - 1)**2))
  rmsw2 = np.sqrt(np.mean((hw2 / h - 1)**2))
  rmswc2 = np.sqrt(np.mean((hwc2 / h - 1)**2))
  rmsw3 = np.sqrt(np.mean((hw3 / h - 1)**2))
  rmswc3 = np.sqrt(np.mean((hwc3 / h - 1)**2))
  #print(f'RMS ({n1}) = {rms1:.5f}, ({n1}/{n} weighted) = {rmsw:.5f}, ({n1}/{n} weighted + corrected) = {rmswcor:.5f}, 2nd iter weighted = {rmsw2:.5f}, + corrected) = {rmswc2:.5f}')
  print(f'RMS ({n1}) = {rms1:.5f}, ({n1}/{n} weighted) = {rmsw:.5f}, ({n1}/{n} weighted + corrected) = {rmswcor:.5f}, 2nd iter weighted = {rmsw2:.5f}, + corrected) = {rmswc2:.5f}, 3rd iter weighted = {rmsw3:.5f}, + corrected) = {rmswc3:.5f}')

  ax = axs[0, iobs]
  ax.stairs(h, bins, label=f'{n} events')
  ax.stairs(h1, bins, label=f'{n1} events')
  ax.stairs(hw, bins, label=f'{n1} events weighted')
  ax.stairs(hwcorr, bins, label=f'{n1} events weighted + cor.')
  ax.stairs(hw2, bins, label=f'{n1} events weighted(2)')
  ax.stairs(hwc2, bins, label=f'{n1} events weighted + cor.(2)')
  ax.stairs(hw3, bins, label=f'{n1} events weighted(3)')
  ax.stairs(hwc3, bins, label=f'{n1} events weighted + cor.(3)')

  ax.set_ylabel('Probability')

  # ratio
  rax = axs[1, iobs]
  rax.stairs(h / h, bins)
  rax.stairs(h1 / h, bins)
  rax.stairs(hw / h, bins)
  rax.stairs(hwcorr / h, bins)
  rax.stairs(hw2 / h, bins)
  rax.stairs(hwc2 / h, bins)
  rax.stairs(hw3 / h, bins)
  rax.stairs(hwc3 / h, bins)
  #rax.axhline(1, color='black', linestyle='--')

  rax.set_xlabel('x')
  rax.set_ylabel('Ratio')
  rax.set_ylim(0.95,1.05)
axs[0,0].legend()

# 2D: rapidity distributions in pT bins
pt_bins = [0, 50, 80, 100, 125, 150, 200, 300]
rap_bins = [-2.1, -1., -0.5, 0.0, 0.5, 1.0, 2.1]

pt = obs_pt(x)
rap = obs_rap(x)

pt1 = obs_pt(x1)
rap1 = obs_rap(x1)

pt1c = obs_pt(x1_corr)
rap1c = obs_rap(x1_corr)

pt2 = obs_pt(x1_corr)
rap2 = obs_rap(x1_corr)

pt2c = obs_pt(x1_corr2)
rap2c = obs_rap(x1_corr2)

pt3 = obs_pt(x1_corr2)
rap3 = obs_rap(x1_corr2)

pt3c = obs_pt(x1_corr3)
rap3c = obs_rap(x1_corr3)

fig, axs = plt.subplots(
    2, len(pt_bins)-1, figsize=(18, 10),
    sharex=1, sharey='row',
    #gridspec_kw={'height_ratios': [3, 1, 3, 1],
    #             'hspace': 0.08}
)
labels = [
    'unweighted',
    'weighted 1',
    'weighted + cor. 1',
    'weighted 2',
    'weighted + cor. 2',
    'weighted 3',
    'weighted + cor. 3'
]
rms = [0.] * len(labels)
for j in range(len(pt_bins)-1):
  ptmin = pt_bins[j]
  ptmax = pt_bins[j + 1]

  # original
  m = (pt >= ptmin) & (pt < ptmax)
  h, _ = np.histogram(rap[m], rap_bins)

  # iteration 1
  m1 = (pt1 >= ptmin) & (pt1 < ptmax)
  h1, _ = np.histogram(
      rap1[m1], rap_bins,
      weights=np.ones_like(rap1[m1]) * n / n1
  )
  hw, _ = np.histogram(
      rap1[m1], rap_bins,
      weights=weights[m1]
  )

  m1c = (pt1c >= ptmin) & (pt1c < ptmax)
  hwcorr, _ = np.histogram(
      rap1c[m1c], rap_bins,
      weights=weights[m1c]
  )

  # iteration 2
  m2 = (pt2 >= ptmin) & (pt2 < ptmax)
  hw2, _ = np.histogram(
      rap2[m2], rap_bins,
      weights=weights2[m2]
  )

  m2c = (pt2c >= ptmin) & (pt2c < ptmax)
  hwc2, _ = np.histogram(
      rap2c[m2c], rap_bins,
      weights=weights2[m2c]
  )

  # iteration 3
  m3 = (pt3 >= ptmin) & (pt3 < ptmax)
  hw3, _ = np.histogram(
      rap3[m3], rap_bins,
      weights=weights3[m3]
  )

  m3c = (pt3c >= ptmin) & (pt3c < ptmax)
  hwc3, _ = np.histogram(
      rap3c[m3c], rap_bins,
      weights=weights3[m3c]
  )

  hs = [h, h1, hw, hwcorr, hw2, hwc2, hw3, hwc3]

  mask = h > 0

  for ihh,hh in enumerate(hs[1:]):
    rms[ihh] += np.sqrt(np.mean((hh[mask] / h[mask] - 1)**2))

  # distributions
  ax = axs[0, j]
  ax.stairs(h, rap_bins, label=f'{n} events')
  ax.stairs(h1, rap_bins, label='unweighted')
  ax.stairs(hw, rap_bins, label='weighted 1')
  ax.stairs(hwcorr, rap_bins, label='weighted + cor. 1')
  ax.stairs(hw2, rap_bins, label='weighted 2')
  ax.stairs(hwc2, rap_bins, label='weighted + cor. 2')
  ax.stairs(hw3, rap_bins, label='weighted 3')
  ax.stairs(hwc3, rap_bins, label='weighted + cor. 3')

  ax.set_title(
      f'$p_T \\in [{ptmin:g}, {ptmax:g}]$'
  )
  # ratio
  rax = axs[1, j]
  for hh in hs:
      ratio = np.divide(
          hh, h,
          out=np.ones_like(h, dtype=float),
          where=h > 0
      )
      rax.stairs(ratio, rap_bins)
  rax.set_ylim(0.95, 1.05)
  rax.set_xlabel('Rapidity')

  axs[0, 0].set_ylabel('Events')
  axs[1, 0].set_ylabel('Ratio')

  axs[0, 0].legend(fontsize=8)

print(
    'RMS: '
    + ', '.join(f'{l} = {r:.5f}' for l, r in zip(labels, rms))
)
plt.tight_layout()
plt.show()