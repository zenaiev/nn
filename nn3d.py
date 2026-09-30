import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

np.random.seed(42)

n = 500000
n1 = 50000
a = 3

x = np.random.exponential(a, (n, 3))

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
def obs(x):
  # pT
  return np.sqrt(x[:,0]**2+x[:,1]**2)
bins = np.linspace(0, 10, 51)
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

fig, (ax, rax) = plt.subplots(
    2, 1, figsize=(8, 6),
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05}
)

ax.stairs(h, bins, label=f'{n} events')
ax.stairs(h1, bins, label=f'{n1} events')
ax.stairs(hw, bins, label=f'{n1} events weighted')
ax.stairs(hwcorr, bins, label=f'{n1} events weighted + cor.')
ax.stairs(hw2, bins, label=f'{n1} events weighted(2)')
ax.stairs(hwc2, bins, label=f'{n1} events weighted + cor.(2)')
ax.stairs(hw3, bins, label=f'{n1} events weighted(3)')
ax.stairs(hwc3, bins, label=f'{n1} events weighted + cor.(3)')

ax.set_ylabel('Probability')
ax.legend()

# ratio
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
rax.set_ylim(0.8,1.2)
plt.show()