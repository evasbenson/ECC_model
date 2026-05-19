import scipy.io as sio
import numpy as np

# load spatial data
data = sio.loadmat('hcc_spatial_input.mat')
PAR       = data['PAR']        # (180, 360, 365) MJ/m2/day
area      = data['area']       # (180, 360) m2
eimat     = data['eimat']      # (180, 360, 365)
farmedPct = data['farmedPct']  # (180, 360)

# technology level to test
tau = 0.5
conversion = tau * (0.123 - 0.123/2.5) + 0.123/2.5
harvest    = 1 - (tau * (0 - 0.5) + 0.5)

# annual food productivity per m2 (kcal/m2/year)
# sum daily NPP over the year, same logic as humanlimits.py
potentialNPP = conversion * eimat * PAR        # MJ/m2/day per day
annualNPP    = np.sum(potentialNPP, axis=2)    # MJ/m2/year, sum over 365 days
kcal_per_m2  = 239.006 * harvest * annualNPP   # kcal/m2/year

# convert to kcal/ha/year (1 ha = 10000 m2)
kcal_per_ha = kcal_per_m2 * 10000

# area in hectares
area_ha = area / 10000

print(f'kcal_per_ha: min={np.nanmin(kcal_per_ha):.0f}, max={np.nanmax(kcal_per_ha):.0f}')
print(f'area_ha: min={np.nanmin(area_ha):.0f}, max={np.nanmax(area_ha):.0f}')
print(f'total farmed area (Gha): {np.nansum(farmedPct * area_ha) / 1e9:.2f}')

def compute_effective_g(kcal_per_ha, area_ha, farmedPct, h_food_share):
    """
    Ranks grid cells by food productivity and returns the
    production-weighted average g for the top h_food_share fraction of land.
    """
    total_ha = np.nansum(area_ha)
    target_ha = h_food_share * total_ha

    # flatten arrays
    g_flat    = kcal_per_ha.ravel()
    area_flat = area_ha.ravel()

    # rank by productivity (highest first)
    order = np.argsort(-g_flat)

    # greedily fill until target hectares reached
    accumulated = 0.0
    weighted_g  = 0.0
    for idx in order:
        if not np.isfinite(g_flat[idx]) or area_flat[idx] <= 0:
            continue
        if accumulated >= target_ha:
            break
        take = min(area_flat[idx], target_ha - accumulated)
        weighted_g  += g_flat[idx] * take
        accumulated += take

    effective_g = weighted_g / accumulated if accumulated > 0 else 0.0
    return effective_g

# test across a range of food land shares
print('\nEffective g vs food land share:')
for share in [0.1, 0.2, 0.3, 0.4, 0.5]:
    g = compute_effective_g(kcal_per_ha, area_ha, farmedPct, share)
    print(f'  h_food_share={share:.1f} -> effective_g={g:.0f} kcal/ha/yr')

# compare to current aspatial value
PAR_scalar  = 2164.5
PAR_kcal    = 239.006 * PAR_scalar * 10000
g_aspatial  = PAR_kcal * harvest * conversion
print(f'\nCurrent aspatial g = {g_aspatial:.0f} kcal/ha/yr')

from ecc_parameters import get_parameters
from ecc_optimization import ecc_optimization

params = get_parameters()
mu0, mu1 = params['mu0'], params['mu1']
gamma0, gamma1, gamma2, gamma3 = params['gamma0'], params['gamma1'], params['gamma2'], params['gamma3']
alpha0, alpha1, alpha2 = params['alpha0'], params['alpha1'], params['alpha2']
eta0, eta1 = params['eta0'], params['eta1']

k_denom = alpha0*mu1 + mu0*gamma0 + eta0*(mu0*gamma2 + mu1*alpha2)
l_denom = alpha1*mu1 + mu0*gamma1 + eta1*(mu0*gamma2 + mu1*alpha2)
n_denom = mu0*gamma2 + mu1*alpha2
h_denom = 1 + alpha2 + mu0*(gamma3 + gamma2 - 1 - alpha2)

sols = np.array([
    gamma0*mu0/k_denom, alpha0*mu1/k_denom, eta0*(mu0*gamma2+mu1*alpha2)/k_denom,
    gamma1*mu0/l_denom, alpha1*mu1/l_denom, eta1*(mu0*gamma2+mu1*alpha2)/l_denom,
    mu0*gamma2/n_denom, mu1*alpha2/n_denom,
    mu0*gamma3/h_denom, mu1/h_denom, (mu0*gamma2+mu1*alpha2)/h_denom
])
params['sols'] = sols

# use spatial effective g
h_food_share = sols[9]  # initial food land share from analytic solution
eff_g = compute_effective_g(kcal_per_ha, area_ha, farmedPct, h_food_share)

# back out effective PAR from g
PAR_KCAL_SCALE = 239.006 * 10000
params['PAR'] = eff_g / (PAR_KCAL_SCALE * harvest * conversion)

# run optimizer
pop, tech = 100, 0.5
L = params['lfpr'] * pop
ece = tech * (0.868 - 0.22) + 0.22
nmax = ece * params['PAR'] * sols[10] * (params['H']/L) * sols[2]**eta0 * sols[5]**eta1
Omega = params['A'] * sols[0]**gamma0 * sols[3]**gamma1 * (sols[6]*nmax)**gamma2 * (sols[8]*(params['H']/L))**gamma3
kmax = (params['s']/params['d'] * Omega)**(1/(1-gamma0))
params['nmax'] = nmax
params['kmax'] = kmax

sol, util, flag = ecc_optimization(pop, tech, params)
print(f'\nSpatial result:')
print(f'exitflag: {flag}')
print(f'utility:  {-util:.6f}')
print(f'h_food_share: {sol[9]:.4f}')

# compare to aspatial
params['PAR'] = 2164.5
nmax = ece * params['PAR'] * sols[10] * (params['H']/L) * sols[2]**eta0 * sols[5]**eta1
Omega = params['A'] * sols[0]**gamma0 * sols[3]**gamma1 * (sols[6]*nmax)**gamma2 * (sols[8]*(params['H']/L))**gamma3
kmax = (params['s']/params['d'] * Omega)**(1/(1-gamma0))
params['nmax'] = nmax
params['kmax'] = kmax

sol2, util2, flag2 = ecc_optimization(pop, tech, params)
print(f'\nAspatial result:')
print(f'exitflag: {flag2}')
print(f'utility:  {-util2:.6f}')
print(f'h_food_share: {sol2[9]:.4f}')