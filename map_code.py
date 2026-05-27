# original land allocation map
h_food_share = sol_s[9]  # optimal food share from spatial result
total_ha  = np.nansum(area_ha)
target_ha = h_food_share * total_ha

g_flat      = kcal_per_ha.ravel()
area_flat   = area_ha.ravel()
farmed_flat = farmedPct.ravel()
land_flat   = is_land.ravel()
order       = np.argsort(-g_flat)

farmed_area   = farmed_flat * area_flat * land_flat
unfarmed_area = (1 - farmed_flat) * area_flat * land_flat
selected      = np.zeros(len(g_flat))
accumulated   = 0.0

for idx in order:
    if farmed_area[idx] <= 0: continue
    if accumulated >= target_ha: break
    take = min(farmed_area[idx], target_ha - accumulated)
    selected[idx] = take / area_flat[idx]
    accumulated  += take

for idx in order:
    if unfarmed_area[idx] <= 0: continue
    if accumulated >= target_ha: break
    take = min(unfarmed_area[idx], target_ha - accumulated)
    selected[idx] = take / area_flat[idx]
    accumulated  += take

selected_2d = np.where(is_land, selected.reshape(180, 360), np.nan)

plt.figure(figsize=(14, 5))
im = plt.pcolormesh(lon2d, lat2d, selected_2d, cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
plt.colorbar(im, label='Fraction allocated to food')
plt.title(f'Land Allocated to Food — h_food_share={h_food_share:.3f}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_facecolor('#aec6e8')
plt.tight_layout()
plt.savefig('land_allocation_map.png', dpi=150, bbox_inches='tight')
plt.show()
print('saved to land_allocation_map.png')

# check what's in the green ocean cells
par_annual = np.mean(PAR_grid, axis=2)
ei_mean    = np.mean(eimat, axis=2)

# check which rows are near the equator so we can include in ocean mask
equator_band = slice(85, 95)  

print('EI values in equatorial band:')
print(f'  min: {ei_mean[equator_band].min():.4f}')
print(f'  max: {ei_mean[equator_band].max():.4f}')
print(f'  mean: {ei_mean[equator_band].mean():.4f}')
print(f'  nonzero cells: {(ei_mean[equator_band] > 0).sum()}')
print()
print('farmedPct in equatorial band:')
print(f'  nonzero cells: {(farmedPct[equator_band] > 0).sum()}')
print(f'  max: {farmedPct[equator_band].max():.4f}')

# binary land allocation map
h_food_share = sol_s[9]

# only farmable cells (farmedPct > 0)
g_flat      = kcal_per_ha.ravel()
area_flat   = area_ha.ravel()
farmed_flat = farmedPct.ravel()
land_flat   = is_land.ravel()
farmable    = (farmed_flat > 0) & land_flat

total_farmable_ha = np.nansum(area_flat * farmable)
target_ha         = min(h_food_share * np.nansum(area_ha * is_land), total_farmable_ha)

order       = np.argsort(-g_flat)
selected    = np.zeros(len(g_flat))
accumulated = 0.0

for idx in order:
    if not farmable[idx] or area_flat[idx] <= 0:
        continue
    if accumulated >= target_ha:
        break
    take = min(area_flat[idx], target_ha - accumulated)
    selected[idx] = take / area_flat[idx]
    accumulated  += take

# ocean = NaN (blue background), unfarmed land = 0 (not shown), farmed = 0-1
selected_2d = np.where(farmedPct > 0, selected.reshape(180, 360), np.nan)

plt.figure(figsize=(14, 5))
im = plt.pcolormesh(lon2d, lat2d, selected_2d, cmap='RdYlGn', 
                    shading='auto', vmin=0, vmax=1)
plt.colorbar(im, label='Fraction allocated to food (green=selected, red=not selected)')
plt.title(f'Land Allocated to Food — h_food_share={h_food_share:.3f}, tau={tau}, pop={pop}B')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_facecolor('#aec6e8')
plt.tight_layout()
plt.savefig('land_allocation_map.png', dpi=150, bbox_inches='tight')
plt.show()
print('saved to land_allocation_map.png')

# test land allocation at different technology level
tau_test = 0.5  
pop_test = 8.0

kcal_per_ha_test, _ = get_kcal_per_ha(PAR_grid, eimat, area, tau_test)

# rerun optimizer
params_test = get_parameters()
sols_test   = get_analytic_sols(params_test)
p_test      = setup_params(pop_test, tau_test, params_test, sols_test, 
                           use_spatial=True,
                           kcal_per_ha=kcal_per_ha_test, area_ha=area_ha,
                           farmedPct=farmedPct, is_land=is_land)
sol_test, util_test, flag_test = ecc_optimization(pop_test, tau_test, p_test)

print(f'tau={tau_test}, pop={pop_test}B')
print(f'exitflag={flag_test}, utility={-util_test:.4f}, h_food={sol_test[9]:.4f}')

# land allocation map
h_food_share_test = sol_test[9]
g_flat_test   = kcal_per_ha_test.ravel()
area_flat     = area_ha.ravel()
farmed_flat   = farmedPct.ravel()
land_flat     = is_land.ravel()
farmable      = (farmed_flat > 0) & land_flat

total_farmable_ha = np.nansum(area_flat * farmable)
target_ha         = min(h_food_share_test * np.nansum(area_ha * is_land), total_farmable_ha)

order       = np.argsort(-g_flat_test)
selected    = np.zeros(len(g_flat_test))
accumulated = 0.0

for idx in order:
    if not farmable[idx] or area_flat[idx] <= 0:
        continue
    if accumulated >= target_ha:
        break
    take = min(area_flat[idx], target_ha - accumulated)
    selected[idx] = take / area_flat[idx]
    accumulated  += take

selected_2d = np.where(farmedPct > 0, selected.reshape(180, 360), np.nan)

plt.figure(figsize=(14, 5))
im = plt.pcolormesh(lon2d, lat2d, selected_2d, cmap='RdYlGn',
                    shading='auto', vmin=0, vmax=1)
plt.colorbar(im, label='Fraction allocated to food (green=selected, red=not selected)')
plt.title(f'Land Allocated to Food — tau={tau_test}, pop={pop_test}B, h_food={h_food_share_test:.3f}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_facecolor('#aec6e8')
plt.tight_layout()
plt.savefig(f'land_allocation_tau{tau_test}.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'saved to land_allocation_tau{tau_test}.png')

# land allocation map at tau=0, but this time not binary and trying to fix ocean masking issue
h_food_share = sol_s[9]
total_ha  = np.nansum(area_ha * is_land)
target_ha = h_food_share * total_ha

g_flat      = kcal_per_ha.ravel()
area_flat   = area_ha.ravel()
farmed_flat = farmedPct.ravel()
land_flat   = is_land.ravel()
order       = np.argsort(-g_flat)

# use farmedPct > 0 as the mask — prevents ocean cells from being selected
farmable      = (farmed_flat > 0) & land_flat
unfarmed_land = (farmed_flat == 0) & land_flat

farmed_area   = farmed_flat * area_flat * farmable
unfarmed_area = area_flat * unfarmed_land

selected    = np.zeros(len(g_flat))
accumulated = 0.0

# first pass: farmed land
for idx in order:
    if farmed_area[idx] <= 0: continue
    if accumulated >= target_ha: break
    take = min(farmed_area[idx], target_ha - accumulated)
    selected[idx] = take / area_flat[idx]
    accumulated  += take

# second pass: unfarmed land only if needed
for idx in order:
    if unfarmed_area[idx] <= 0: continue
    if accumulated >= target_ha: break
    take = min(unfarmed_area[idx], target_ha - accumulated)
    selected[idx] = take / area_flat[idx]
    accumulated  += take

# mask ocean as NaN — only show land cells
selected_2d = np.where(farmedPct > 0, selected.reshape(180, 360), np.nan)

plt.figure(figsize=(14, 5))
im = plt.pcolormesh(lon2d, lat2d, selected_2d, cmap='RdYlGn', shading='auto', vmin=0, vmax=1)
plt.colorbar(im, label='Fraction allocated to food (green=selected, red=not selected)')
plt.title(f'Land Allocated to Food — h_food_share={h_food_share:.3f}, tau={tau}, pop={pop}B')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_facecolor('#aec6e8')
plt.tight_layout()
plt.savefig('land_allocation_map.png', dpi=150, bbox_inches='tight')
plt.show()
print('saved to land_allocation_map.png')

# test higher population to see if cutoff disappears and more land is used
tau_test = 0.0
pop_test = 500.0  

kcal_per_ha_test, _ = get_kcal_per_ha(PAR_grid, eimat, area, tau_test)

# rerun optimizer
params_test = get_parameters()
sols_test   = get_analytic_sols(params_test)
p_test      = setup_params(pop_test, tau_test, params_test, sols_test,
                           use_spatial=True,
                           kcal_per_ha=kcal_per_ha_test, area_ha=area_ha,
                           farmedPct=farmedPct, is_land=is_land)
sol_test, util_test, flag_test = ecc_optimization(pop_test, tau_test, p_test)
h_food_share_test = sol_test[9]

print(f'tau={tau_test}, pop={pop_test}B')
print(f'exitflag={flag_test}, utility={-util_test:.4f}, h_food={h_food_share_test:.4f}')

g_flat_test  = kcal_per_ha_test.ravel()
area_flat    = area_ha.ravel()
farmed_flat  = farmedPct.ravel()
land_flat    = is_land.ravel()
farmable     = (farmed_flat > 0) & land_flat
unfarmed_land = (farmed_flat == 0) & land_flat

total_ha_test = np.nansum(area_flat * land_flat)
target_ha     = h_food_share_test * total_ha_test

farmed_area   = farmed_flat * area_flat * farmable
unfarmed_area = area_flat * unfarmed_land

order       = np.argsort(-g_flat_test)
selected    = np.zeros(len(g_flat_test))
accumulated = 0.0

for idx in order:
    if farmed_area[idx] <= 0: continue
    if accumulated >= target_ha: break
    take = min(farmed_area[idx], target_ha - accumulated)
    selected[idx] = take / area_flat[idx]
    accumulated  += take

for idx in order:
    if unfarmed_area[idx] <= 0: continue
    if accumulated >= target_ha: break
    take = min(unfarmed_area[idx], target_ha - accumulated)
    selected[idx] = take / area_flat[idx]
    accumulated  += take

selected_2d   = selected.reshape(180, 360)
display_2d    = np.where(farmedPct > 0, selected_2d,
                np.where(is_land,      -0.05, np.nan))

from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import matplotlib.colors as mcolors

colors_list = ['#d3d3d3'] + list(plt.cm.RdYlGn(np.linspace(0, 1, 256)))
custom_cmap = LinearSegmentedColormap.from_list('grey_RdYlGn', colors_list, N=257)

plt.figure(figsize=(14, 5))
im = plt.pcolormesh(lon2d, lat2d, display_2d, cmap=custom_cmap,
                    shading='auto', vmin=-0.05, vmax=1)
cbar = plt.colorbar(im, label='Fraction allocated to food')
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(['0 (not selected)', '0.25', '0.5', '0.75', '1.0 (fully selected)'])
plt.title(f'Land Allocated to Food — tau={tau_test}, pop={pop_test}B, h_food={h_food_share_test:.3f}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_facecolor('#aec6e8')
plt.tight_layout()
plt.savefig(f'land_allocation_tau{tau_test}_pop{int(pop_test)}.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'saved to land_allocation_tau{tau_test}_pop{int(pop_test)}.png')
print(f'total selected area: {np.nansum(selected * area_flat)/1e9:.2f} Gha')
print(f'total farmable area: {np.nansum(area_flat * farmable)/1e9:.2f} Gha')