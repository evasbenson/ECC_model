"""
ecc_opt_wrapper.py

Parallelized version of the ECC simulation wrapper.
Uses multiprocessing to solve (tau, pop) grid points simultaneously,
one per CPU core. All other logic is identical to the original.
"""

import numpy as np
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
from ecc_parameters import get_parameters
from ecc_optimization import ecc_optimization


def _solve_one(args):
    """
    Solves a single (tau, pop) grid point.
    This is the function each worker process runs.
    Must be a top-level function (not nested) for multiprocessing to work.

    args is a tuple: (k, j, tech, population, parameters)
    returns: (k, j, solution, utility, kmax)
    """
    k, j, tech, population, parameters = args

    lfpr = parameters['lfpr']
    H = parameters['H']
    s = parameters['s']
    d = parameters['d']
    A = parameters['A']
    PAR = parameters['PAR']
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

    # compute initial kmax and nmax from analytic solution
    ece = tech * (0.868 - 0.22) + 0.22
    nmax = (ece * PAR * sols_analytic[10] * (H / L)
            * sols_analytic[2]**eta0 * sols_analytic[5]**eta1)
    Omega = (A * sols_analytic[0]**gamma0 * sols_analytic[3]**gamma1
             * (sols_analytic[6] * nmax)**gamma2
             * (sols_analytic[8] * (H / L))**gamma3)
    kmax = (s / d * Omega) ** (1 / (1 - gamma0))

    parameters = parameters.copy()
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

        # compute implied kmax and nmax from solution
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


def run_ecc_simulation(tau=None, pop=None, max_iter=200, n_workers=None):
    """
    Runs the full ECC simulation across a technology x population grid,
    using multiprocessing to solve grid points in parallel.

    args:
        tau: array of technology levels (default: 1.0 to 0.0 in steps of 0.01)
        pop: array of population levels (default: 1 to 1201 in steps of 10)
        max_iter: max fixed-point iterations per problem (default: 200)
        n_workers: number of parallel workers (default: all CPU cores)

    returns:
        utilmat, solutioncube, tau, pop, parameters
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
    k_denom = alpha0 * mu1 + mu0 * gamma0 + eta0 * (mu0 * gamma2 + mu1 * alpha2)
    l_denom = alpha1 * mu1 + mu0 * gamma1 + eta1 * (mu0 * gamma2 + mu1 * alpha2)
    n_denom = mu0 * gamma2 + mu1 * alpha2
    h_denom = 1 + alpha2 + mu0 * (gamma3 + gamma2 - 1 - alpha2)

    sols_analytic = np.array([
        gamma0 * mu0 / k_denom,
        alpha0 * mu1 / k_denom,
        eta0 * (mu0 * gamma2 + mu1 * alpha2) / k_denom,
        gamma1 * mu0 / l_denom,
        alpha1 * mu1 / l_denom,
        eta1 * (mu0 * gamma2 + mu1 * alpha2) / l_denom,
        mu0 * gamma2 / n_denom,
        mu1 * alpha2 / n_denom,
        mu0 * gamma3 / h_denom,
        mu1 / h_denom,
        (mu0 * gamma2 + mu1 * alpha2) / h_denom
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

    # number of workers
    if n_workers is None:
        n_workers = cpu_count()
    print(f"grid size: {n_tau} tech x {n_pop} pop = {total} problems")
    print(f"running with {n_workers} parallel workers")

    # build list of all jobs
    jobs = [
        (k, j, tau[k], pop[j], parameters)
        for k in range(n_tau)
        for j in range(n_pop)
    ]

    # initialize result arrays
    utilmat = np.zeros((n_tau, n_pop))
    solutioncube = np.zeros((12, n_tau, n_pop))

    # run in parallel
    # completed = 0
    # with Pool(processes=n_workers) as pool:
        # for k, j, result, utiltemp, kmax in pool.imap_unordered(_solve_one, jobs, chunksize=10):
            # utilmat[k, j] = utiltemp
            # solutioncube[:, k, j] = result
            # completed += 1
            # if completed % 50 == 0 or completed == total:
                # print(f"  {completed}/{total} problems done", flush=True)
    
    # run sequentially
    completed = 0
    for job in jobs:
        k, j, result, utiltemp, kmax = _solve_one(job)
        utilmat[k, j] = utiltemp
        solutioncube[:, k, j] = result
        completed += 1
        print(f"  {completed}/{total} done (tau={tau[k]:.1f}, pop={pop[j]:.0f})", flush=True)

    # flip sign: optimizer minimizes negative utility
    utilmat = -utilmat

    # save results
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"ECC_solutions_{timestamp}.npz"
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
    run_ecc_simulation(
        tau=np.arange(1, 0, -0.1),
        pop=np.arange(1, 202, 20),
        max_iter=10,
        n_workers=1
    )

    # full grid - uncomment when ready
    # run_ecc_simulation()
