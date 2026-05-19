import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from spatial_g import get_kcal_per_ha, compute_effective_g

# load data
data = sio.loadmat('hcc_spatial_input.mat')
PAR       = data['PAR']
area      = data['area']
farmedPct = data['farmedPct']
eimat     = data['eimat']
area_ha   = area / 10000

# compute productivity at tau=0.5
tau = 0.5
kcal_per_ha, _ = get_kcal_per_ha(PAR, eimat, area, tau)

# lat/lon grid (1 degree resolution, per figshare description)
lon = np.linspace(-179.5, 179.5, 360)
lat = np.linspace(89.5, -89.5, 180)
lon2d, lat2d = np.meshgrid(lon, lat)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# --- map 1: food productivity ---
ax = axes[0]
im = ax.pcolormesh(lon2d, lat2d, kcal_per_ha/1e6,
                   cmap='YlGn', shading='auto', vmin=0, vmax=200)
plt.colorbar(im, ax=ax, label='kcal/ha/yr (millions)')
ax.set_title(f'Food Productivity (tau={tau})')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_facecolor('#cccccc')

# --- map 2: land selected for food at optimal share ---
h_food_share = 0.41  # from spatial optimization result
total_ha = np.nansum(area_ha)
target_ha = h_food_share * total_ha

g_flat    = kcal_per_ha.ravel()
area_flat = area_ha.ravel()
order     = np.argsort(-g_flat)

selected = np.zeros(len(g_flat))
accumulated = 0.0
for idx in order:
    if not np.isfinite(g_flat[idx]) or area_flat[idx] <= 0:
        continue
    if accumulated >= target_ha:
        break
    take = min(area_flat[idx], target_ha - accumulated)
    selected[idx] = take / area_flat[idx]
    accumulated += take

selected_2d = selected.reshape(180, 360)

ax = axes[1]
im2 = ax.pcolormesh(lon2d, lat2d, selected_2d,
                    cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
plt.colorbar(im2, ax=ax, label='Fraction allocated to food')
ax.set_title(f'Land Allocated to Food (h_food_share={h_food_share})')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_facecolor('#cccccc')

plt.tight_layout()
plt.savefig('spatial_maps.png', dpi=150, bbox_inches='tight')
plt.show()
print('saved to spatial_maps.png')