"""
translated from ECC_opt_wrapper.m
main simulation script that runs the optimization across a grid of
technology levels and population sizes. uses the same fixed-point 
iteration loop as the matlab version to solve for equilibrium kmax and nmax.
"""

import numpy as np
import warnings
from datetime import datetime
from ecc_parameters import get_parameters
from ecc_optimization import ecc_optimization


def run_ecc_simulation(tau=None, pop=None, max_iter=200):
    """
    runs the full ECC simulation across a technology x population grid.
    this is a direct translation of the matlab script ECC_opt_wrapper.m.

    the outer loop iterates over (tau, pop) pairs. for each pair, there's
    a fixed-point iteration that updates kmax and nmax until they converge,
    because capital stock and energy capacity depend on production which 
    depends on capital and energy (circular dependency).

    args:
        tau: array of technology levels (default: 1.0 to 0.0 in steps of 0.01)
        pop: array of population levels (default: 1 to 1201 in steps of 10)
        max_iter: max fixed-point iterations (default: 200, same as matlab)

    returns:
        utilmat: utility matrix (len(tau) x len(pop))
        solutioncube: solution array (12 x len(tau) x len(pop))
        tau: technology levels used
        pop: population levels used
        parameters: parameter dict
    """

    # load parameters (equivalent to running ECC_parameters in matlab)
    parameters = get_parameters()

    # extract parameters we need for the analytic solution and iteration
    lfpr = parameters['lfpr']
    H = parameters['H']
    mu0 = parameters['mu0']
    mu1 = parameters['mu1']
    s = parameters['s']
    d = parameters['d']
    A = parameters['A']
    gamma0 = parameters['gamma0']
    gamma1 = parameters['gamma1']
    gamma2 = parameters['gamma2']
    gamma3 = parameters['gamma3']
    PAR = parameters['PAR']
    alpha0 = parameters['alpha0']
    alpha1 = parameters['alpha1']
    alpha2 = parameters['alpha2']
    eta0 = parameters['eta0']
    eta1 = parameters['eta1']

    # compute the analytic solution for the unconstrained optimum
    # this is used as the initial guess for the numerical optimizer
    # (same formulas as the matlab wrapper)
    k_denom = alpha0 * mu1 + mu0 * gamma0 + eta0 * (mu0 * gamma2 + mu1 * alpha2)
    l_denom = alpha1 * mu1 + mu0 * gamma1 + eta1 * (mu0 * gamma2 + mu1 * alpha2)
    n_denom = mu0 * gamma2 + mu1 * alpha2
    h_denom = 1 + alpha2 + mu0 * (gamma3 + gamma2 - 1 - alpha2)

    sols_analytic = np.array([
        gamma0 * mu0 / k_denom,                          # k_y / k
        alpha0 * mu1 / k_denom,                          # k_f / k
        eta0 * (mu0 * gamma2 + mu1 * alpha2) / k_denom,  # k_n / k
        gamma1 * mu0 / l_denom,                          # l_y
        alpha1 * mu1 / l_denom,                          # l_f
        eta1 * (mu0 * gamma2 + mu1 * alpha2) / l_denom,  # l_n
        mu0 * gamma2 / n_denom,                          # n_y / n
        mu1 * alpha2 / n_denom,                          # n_f / n
        mu0 * gamma3 / h_denom,                          # h_y / h
        mu1 / h_denom,                                   # h_f / h
        (mu0 * gamma2 + mu1 * alpha2) / h_denom          # h_n / h
    ])

    parameters['sols'] = sols_analytic

    # set up the parameter grid (same defaults as matlab)
    # matlab: tau = 1:-0.01:0 and pop = 1:10:1201
    if tau is None:
        tau = np.arange(1, -0.01, -0.01)  # 101 values from 1.0 to 0.0
        # clean up floating point to get exact values
        tau = np.round(tau, 2)
    else:
        tau = np.asarray(tau, dtype=float)
    if pop is None:
        pop = np.arange(1, 1202, 10)  # 121 values from 1 to 1201
    else:
        pop = np.asarray(pop, dtype=float)

    print(f"grid size: {len(tau)} tech x {len(pop)} pop = {len(tau) * len(pop)} problems")

    # initialize result arrays (same shapes as matlab)
    utilmat = np.zeros((len(tau), len(pop)))
    solutioncube = np.zeros((len(sols_analytic) + 1, len(tau), len(pop)))  # 12 x tau x pop

    # fixed-point iteration parameters (same as matlab)
    damping = 0.6

    total = len(tau) * len(pop)
    count = 0

    # main loop - same structure as the matlab nested for loops
    for k in range(len(tau)):
        for j in range(len(pop)):
            tech = tau[k]
            population = pop[j]
            L = lfpr * population
            print(f"starting tau={tech:.2f}, pop={population}", flush=True)

            # compute initial guess for kmax and nmax from the analytic solution
            # (same formulas as matlab)
            ece = tech * (0.868 - 0.22) + 0.22
            nmax = (ece * PAR * sols_analytic[10] * (H / L)
                    * sols_analytic[2]**eta0 * sols_analytic[5]**eta1)
            Omega = (A * sols_analytic[0]**gamma0 * sols_analytic[3]**gamma1
                     * (sols_analytic[6] * nmax)**gamma2
                     * (sols_analytic[8] * (H / L))**gamma3)
            kmax = (s / d * Omega) ** (1 / (1 - gamma0))
            parameters['nmax'] = nmax
            parameters['kmax'] = kmax

            # tolerance depends on population size (same logic as matlab)
            if L <= 1:
                tolerance = 1e-4
            else:
                scale_factor = max(1, population / 25)
                tolerance = 1e-5 * scale_factor

            # fixed-point iteration loop (same as matlab)
            for iter_num in range(max_iter):
                try:
                    soltemp, utiltemp, exitflag = ecc_optimization(
                        population, tech, parameters
                    )
                except Exception as e:
                    warnings.warn(f"exception at tau={tech:.2f}, pop={population:.1f}: {e}")
                    utilmat[k, j] = -np.inf
                    solutioncube[:, k, j] = -np.inf
                    break

                if exitflag == -2:
                    # infeasible: caloric constraint can't be satisfied
                    utilmat[k, j] = np.nan
                    solutioncube[:, k, j] = np.nan
                    break
                elif exitflag <= 0:
                    # solver failure
                    warnings.warn(
                        f"solver failed (exitflag={exitflag}) at "
                        f"tau={tech:.2f}, pop={population:.1f}, iter={iter_num}"
                    )
                    utilmat[k, j] = -np.inf
                    solutioncube[:, k, j] = -np.inf
                    break

                # calculate implied kmax and nmax from the solution
                # (same formulas as matlab)
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

                # check convergence
                if error_k < tolerance and error_n < tolerance:
                    utilmat[k, j] = utiltemp
                    solutioncube[0:11, k, j] = soltemp
                    solutioncube[11, k, j] = kmax
                    break

                # update with damping (same as matlab)
                parameters['kmax'] = damping * kmax_implied + (1 - damping) * parameters['kmax']
                parameters['nmax'] = damping * nmax_implied + (1 - damping) * parameters['nmax']

                if iter_num == max_iter - 1:
                    warnings.warn(
                        f"fixed-point did not converge for tau={tech:.2f}, pop={population:.1f}"
                    )
                    utilmat[k, j] = np.inf
                    solutioncube[:, k, j] = np.inf

            count += 1
            if count % 10 == 0 or count == total:
                print(f"  {count}/{total} problems done")

    # flip sign back: we stored negative utility from the minimizer
    # (same as matlab's utilmat = -utilmat at the end)
    utilmat = -utilmat

    # save results to a timestamped npz file (equivalent to matlab's .mat save)
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
        tau=np.arange(1, 0, -0.1),   # 10 values instead of 101
        pop=np.arange(1, 202, 20),    # 11 values instead of 121
        max_iter=50                    # fewer iterations
    )
    
