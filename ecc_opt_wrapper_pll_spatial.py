"""
ecc_opt_wrapper_pll.py

ECC simulation wrapper with spatial land allocation.
Replaces scalar PAR with spatially-derived effective g
computed from the HCC spatial input data.
"""

import numpy as np
from datetime import datetime
from ecc_parameters import get_parameters
from ecc_optimization import ecc_optimization
from spatial_g import load_spatial_data, compute_effective_g, get_kcal_per_ha


def _solve_one(args):
    """
    Solves a single (tau, pop) grid point with spatial g.

    args: (k, j, tech, population, parameters, kcal_per_ha, area_ha)
    returns: (k, j, solution array, utility, kmax)
    """
    k, j, tech, population, parameters, kcal_per_ha, area_ha = args

    lfpr = parameters['lfpr']
    H = parameters['H']
    s = parameters['s']
    d = parameters['d']
    A = parameters['A']
    gamma0 = parameters['gamma0']
    gamma1 = parameters['gamma1']
    gamma2 = parameters['gamma2']
    gamma3 = parameters['gamma3']
    eta0 = parameters['eta0']
    eta1 = parameters['eta1']
    sols_analytic = parameters['sols']
    damping = 0.6
    max_iter = parameters.get('max_iter', 200)

    L = lfpr * population
    ece = tech * (0.868 - 0.22) + 0.22

    # get technology scaling for g
    conversion = tech * (0.123 - 0.123/2.5) + 0.123/2.5
    harvest    = 1 - (tech * (0 - 0.5) + 0.5)
    PAR_KCAL_SCALE = 239.006 * 10000

    parameters = parameters.copy()

    # compute initial kmax/nmax using analytic food share as initial guess
    h_food_share = sols_analytic[9]
    eff_g = compute_effective_g(kcal_per_ha, area_ha, h_food_share)
    parameters['PAR'] = eff_g / (PAR_KCAL_SCALE * harvest * conversion) if (harvest * conversion) > 0 else parameters['PAR']

    PAR = parameters['PAR']
    nmax = (ece * PAR * sols_analytic[10] * (H / L)
            * sols_analytic[2]**eta0 * sols_analytic[5]**eta1)
    Omega = (A * sols_analytic[0]**gamma0 * sols_analytic[3]**gamma1
             * (sols_analytic[6] * nmax)**gamma2
             * (sols_analytic[8] * (H / L))**gamma3)
    kmax = (s / d * Omega) ** (1 / (1 - gamma0))

    parameters['nmax'] = nmax
    parameters['kmax'] = kmax

    # tolerance scales with population
    if L <= 1:
        tolerance = 1e-4
    else:
        scale_factor = max(1, population / 25)
        tolerance = 1e-5 * scale_factor

    # fixed-point iteration
    for iter_num in range(max_iter):
        try:
            soltemp, utiltemp, exitflag = ecc_optimization(
                population, tech, parameters
            )
        except Exception:
            return k, j, np.full(12, -np.inf), -np.inf, kmax

        if exitflag == -2:
            return k, j, np.full(12, np.nan), np.nan, kmax
        elif exitflag <= 0:
            return k, j, np.full(12, -np.inf), -np.inf, kmax

        # update effective g based on current food land share
        h_food_share = soltemp[9]
        eff_g = compute_effective_g(kcal_per_ha, area_ha, h_food_share)
        parameters['PAR'] = eff_g / (PAR_KCAL_SCALE * harvest * conversion) if (harvest * conversion) > 0 else parameters['PAR']
        PAR = parameters['PAR']

        # compute implied kmax and nmax
        y_implied = (A * (soltemp[0] * kmax)**gamma0
                     * soltemp[3]**gamma1
                     * (soltemp[6] * nmax)**gamma2
                     * (soltemp[8] * (H / L))**gamma3)
        kmax_implied = s / d * y_implied
        nmax_implied = (ece * PAR * soltemp[10] * (H / L)
                        * (soltemp[2] * kmax / kmax_implied)**eta0
                        * soltemp[5]**eta1)

        error_k = abs(kmax_implied - parameters['kmax'])
        error_n = abs(nmax_implied - parameters['nmax'])

        if error_k < tolerance and error_n < tolerance:
            result = np.zeros(12)
            result[0:11] = soltemp
            result[11] = kmax
            return k, j, result, utiltemp, kmax

        # damped update
        parameters['kmax'] = damping * kmax_implied + (1 - damping) * parameters['kmax']
        parameters['nmax'] = damping * nmax_implied + (1 - damping) * parameters['nmax']
        kmax = parameters['kmax']
        nmax = parameters['nmax']

    # did not converge
    return k, j, np.full(12, np.inf), np.inf, kmax


def run_ecc_simulation(tau=None, pop=None, max_iter=200):
    """
    Runs the full ECC simulation with spatial land allocation.
    Sequential execution (reliable on Mac).
    """
    parameters = get_parameters()

    mu0 = parameters['mu0']
    mu1 = parameters['mu1']
    gamma0 = parameters['gamma0']
    gamma1 = parameters['gamma1']
    gamma2 = parameters['gamma2']
    gamma3 = parameters['gamma3']
    alpha0 = parameters['alpha0']
    alpha1 = parameters['alpha1']
    alpha2 = parameters['alpha2']
    eta0 = parameters['eta0']
    eta1 = parameters['eta1']

    # analytic solution for initial guess
    k_denom = alpha0*mu1 + mu0*gamma0 + eta0*(mu0*gamma2 + mu1*alpha2)
    l_denom = alpha1*mu1 + mu0*gamma1 + eta1*(mu0*gamma2 + mu1*alpha2)
    n_denom = mu0*gamma2 + mu1*alpha2
    h_denom = 1 + alpha2 + mu0*(gamma3 + gamma2 - 1 - alpha2)

    sols_analytic = np.array([
        gamma0*mu0/k_denom, alpha0*mu1/k_denom, eta0*(mu0*gamma2+mu1*alpha2)/k_denom,
        gamma1*mu0/l_denom, alpha1*mu1/l_denom, eta1*(mu0*gamma2+mu1*alpha2)/l_denom,
        mu0*gamma2/n_denom, mu1*alpha2/n_denom,
        mu0*gamma3/h_denom, mu1/h_denom, (mu0*gamma2+mu1*alpha2)/h_denom
    ])

    parameters['sols'] = sols_analytic
    parameters['max_iter'] = max_iter

    # set up grid
    if tau is None:
        tau = np.round(np.arange(1, -0.01, -0.01), 2)
    else:
        tau = np.asarray(tau, dtype=float)
    if pop is None:
        pop = np.arange(1, 1202, 10)
    else:
        pop = np.asarray(pop, dtype=float)

    n_tau = len(tau)
    n_pop = len(pop)
    total = n_tau * n_pop

    print(f"grid size: {n_tau} tech x {n_pop} pop = {total} problems")
    print("loading spatial data...")

    # load spatial data once — shared across all problems
    PAR_grid, area, farmedPct, eimat = load_spatial_data()
    area_ha = area / 10000

    # initialize result arrays
    utilmat = np.zeros((n_tau, n_pop))
    solutioncube = np.zeros((12, n_tau, n_pop))

    # run sequentially
    completed = 0
    for k in range(n_tau):
        tech = tau[k]

        # compute kcal_per_ha for this technology level
        kcal_per_ha, _ = get_kcal_per_ha(PAR_grid, eimat, area, tech)

        for j in range(n_pop):
            population = pop[j]
            args = (k, j, tech, population, parameters, kcal_per_ha, area_ha)
            k_out, j_out, result, utiltemp, kmax = _solve_one(args)
            utilmat[k_out, j_out] = utiltemp
            solutioncube[:, k_out, j_out] = result
            completed += 1
            print(f"  {completed}/{total} done (tau={tech:.1f}, pop={population:.0f})", flush=True)

    # flip sign
    utilmat = -utilmat

    # save results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"ECC_solutions_spatial_{timestamp}.npz"
    np.savez_compressed(
        filename,
        solutioncube=solutioncube,
        utilmat=utilmat,
        tau=tau,
        pop=pop,
        parameters=parameters
    )
    print(f"results saved to {filename}")

    return utilmat, solutioncube, tau, pop, parameters


if __name__ == "__main__":
    # small test grid
    run_ecc_simulation(
        tau=np.arange(1, 0, -0.1),
        pop=np.arange(1, 202, 20),
        max_iter=10
    )

    # full grid - uncomment when ready
    # run_ecc_simulation()