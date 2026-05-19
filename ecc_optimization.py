"""
translated from ECC_optimization.m
uses cyipopt (IPOPT solver) which is the closest python equivalent to 
matlab's fmincon with the interior-point algorithm. both are primal-dual 
interior-point barrier methods so the results should be very close.
"""

import numpy as np
from cyipopt import minimize_ipopt


def ecc_optimization(population, technology, parameters):
    """
    solves the constrained utility maximization for a given population and
    technology level. this is a direct translation of the matlab function
    that uses fmincon with interior-point.

    the decision variables x are 11 shares/fractions that describe how
    capital, labor, energy, and land are allocated across three sectors:
    aggregate goods (y), food (f), and energy (n).

    x = [k_y/k, k_f/k, k_n/k, l_y, l_f, l_n, n_y/n, n_f/n, h_y/h, h_f/h, h_n/h]

    args:
        population: population in billions
        technology: tech level between 0 (current) and 1 (maximal)
        parameters: dict of model parameters from ecc_parameters.py

    returns:
        equilibrium: optimal allocation array (11 elements)
        negutility: negative of utility at optimum (since we minimize)
        exitflag: 1=success, -2=infeasible, 0=max iter, -1=failure
    """

    # set population, technology, and labor
    # same as the matlab function preamble
    pop = population
    tau = technology
    lfpr = parameters['lfpr']
    H = parameters['H']
    L = lfpr * pop  # total labor endowment in billions

    # utility parameters
    mu0 = parameters['mu0']
    mu1 = parameters['mu1']

    # calorie requirement
    kcalmin = parameters['kcalmin']

    # savings rate
    s = parameters['s']

    # aggregate production function parameters
    A = parameters['A']
    gamma0 = parameters['gamma0']
    gamma1 = parameters['gamma1']
    gamma2 = parameters['gamma2']
    gamma3 = parameters['gamma3']

    # food production parameters
    PAR = parameters['PAR']
    PAR_kcal = 2390060 * PAR  # converts PAR to kcal per hectare
    alpha0 = parameters['alpha0']
    alpha1 = parameters['alpha1']
    alpha2 = parameters['alpha2']

    # energy production parameters
    eta0 = parameters['eta0']
    eta1 = parameters['eta1']
    ece = tau * (0.868 - 0.22) + 0.22  # solar energy conversion efficiency

    # scaling parameters that come from the fixed-point iteration in the wrapper
    sols_analytic = parameters['sols']
    kmax = parameters['kmax']
    nmax = parameters['nmax']

    # -------------------------------------------------------------------------
    # objective function: negative utility for minimization
    # direct translation of the nested utility(x,tau) function in the matlab,
    # wrapped as objfunhandle = @(x) -utility(x,tau)
    # -------------------------------------------------------------------------
    def objective(x):
        # unpack control variables exactly like the matlab code
        k_y = x[0] * kmax       # capital for aggregate good
        l_y = x[3]              # labor for aggregate good
        n_y = x[6] * nmax       # energy for aggregate good
        h_y = x[8] * (H / L)   # land for aggregate good

        k_f = x[1] * kmax       # capital for food
        l_f = x[4]              # labor for food
        n_f = x[7] * nmax       # energy for food
        h_f = x[9] * (H / L)   # land for food

        # set technical efficiency parameters (depends on technology level)
        conversion = tau * (0.123 - 0.123 / 2.5) + 0.123 / 2.5  # PAR to biomass
        harvest = 1 - (tau * (0 - 0.5) + 0.5)  # harvestable fraction
        g = PAR_kcal * harvest * conversion  # max kcal production per hectare

        # calculate aggregate production (cobb-douglas)
        y = A * k_y**gamma0 * l_y**gamma1 * n_y**gamma2 * h_y**gamma3

        # consumption per worker then per capita
        c = (1 - s) * y
        c_pc = c * lfpr

        # food production per worker then per capita
        f = g * h_f * (k_f / kmax)**alpha0 * l_f**alpha1 * (n_f / nmax)**alpha2
        f_pc = f * lfpr

        # utility function: u = c_pc^mu0 * f_pc^mu1
        u = c_pc**mu0 * f_pc**mu1
        return -u

    # -------------------------------------------------------------------------
    # constraint functions
    # translated from the nested nonlincon(x) function in the matlab code
    # matlab's nonlincon returns [c, ceq] where c <= 0 and ceq = 0
    # for ipopt we split them into separate dicts with the right sign conventions
    # -------------------------------------------------------------------------
    def energy_balance(x):
        """equality constraint: energy produced = energy used
        this is ceq in the matlab code: ceq = n_y + n_f - n"""
        k_n = x[2] * kmax
        l_n = x[5]
        n_y = x[6] * nmax
        n_f = x[7] * nmax
        h_n = x[10] * (H / L)

        # energy production per worker
        n = ece * PAR * h_n * (k_n / kmax)**eta0 * l_n**eta1
        return n_y + n_f - n

    def calorie_surplus(x):
        """inequality constraint: must produce enough calories
        matlab has c = kcalmin - f_pc (where c <= 0)
        we flip it to f_pc - kcalmin >= 0 for ipopt's convention"""
        k_f = x[1] * kmax
        l_f = x[4]
        n_f = x[7] * nmax
        h_f = x[9] * (H / L)

        conversion = tau * (0.123 - 0.123 / 2.5) + 0.123 / 2.5
        harvest = 1 - (tau * (0 - 0.5) + 0.5)
        g = PAR_kcal * harvest * conversion

        f = g * h_f * (k_f / kmax)**alpha0 * l_f**alpha1 * (n_f / nmax)**alpha2
        f_pc = f * lfpr
        return f_pc - kcalmin

    # set up constraints for ipopt
    # the linear equalities match matlab's Aeq * x = beq (capital, labor, land sum to 1)
    # the nonlinear constraints match matlab's nonlincon function
    constraints = [
        {'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - 1.0},   # capital adding-up
        {'type': 'eq', 'fun': lambda x: x[3] + x[4] + x[5] - 1.0},   # labor adding-up
        {'type': 'eq', 'fun': lambda x: x[8] + x[9] + x[10] - 1.0},  # land adding-up
        {'type': 'eq', 'fun': energy_balance},                          # energy balance
        {'type': 'ineq', 'fun': calorie_surplus},                       # calorie floor
    ]

    # variable bounds: lb = 0.0001, ub = 0.9999 (same as matlab)
    bounds = [(0.0001, 0.9999)] * 11

    # initial guess from analytic solution (same as matlab's x0 = sols_analytic)
    x0 = np.clip(sols_analytic.copy(), 0.0001, 0.9999)

    # ipopt options chosen to match matlab's fmincon interior-point settings as closely
    # as possible. both solvers are interior-point barrier methods so these map well.
    # the matlab code uses tighter tolerances for larger populations (L >= 2)
    if L < 2:
        options = {
            'print_level': 0,                       # like Display='off' in matlab
            'sb': 'yes',                             # suppress the ipopt startup banner
            'max_iter': 3000,                        # matches MaxIterations=3000
            'tol': 1e-7,                             # matches OptimalityTolerance=1e-7
            'constr_viol_tol': 1e-7,                 # matches ConstraintTolerance=1e-7
            'acceptable_tol': 1e-6,
            'acceptable_constr_viol_tol': 1e-6,
            'nlp_scaling_method': 'gradient-based',  # matches ScaleProblem=true
            'mu_strategy': 'adaptive',               # adaptive barrier parameter
        }
    else:
        options = {
            'print_level': 0,
            'sb': 'yes',
            'max_iter': 2000,                        # matches MaxIterations=2000
            'tol': 1e-8,                             # matches OptimalityTolerance=1e-8
            'constr_viol_tol': 1e-8,                 # matches ConstraintTolerance=1e-8
            'acceptable_tol': 1e-7,
            'acceptable_constr_viol_tol': 1e-7,
            'nlp_scaling_method': 'gradient-based',
            'mu_strategy': 'adaptive',
        }

    # solve using ipopt (this is the equivalent of the fmincon call in matlab)
    try:
        result = minimize_ipopt(
            objective, x0,
            bounds=bounds,
            constraints=constraints,
            options=options
        )
        exitflag = _map_exit_flag(result, energy_balance, calorie_surplus)
        return result.x, result.fun, exitflag
    except Exception:
        # if ipopt completely crashes, return the initial guess with failure flag
        return x0, objective(x0), -1


def _map_exit_flag(result, energy_fn, calorie_fn):
    """
    maps ipopt result to matlab fmincon-style exit flags.

    instead of just trusting the ipopt status code, we check whether the
    solution actually satisfies the constraints. ipopt can return non-zero
    status codes (like -2=restoration_failed, 3=search_direction_too_small)
    even when the solution is perfectly usable. matlab's fmincon is more
    lenient in what it considers "success", so we check constraints directly.

    returns:
        1  = success (constraints satisfied, objective finite)
        -2 = infeasible (caloric constraint violated)
        0  = max iterations exceeded
        -1 = genuine failure (constraints not satisfied)
    """
    x = result.x
    status = getattr(result, 'status', -100)

    # check if the solution actually satisfies constraints
    # this is more robust than relying on ipopt's status code alone
    try:
        k_sum_err = abs(x[0] + x[1] + x[2] - 1.0)
        l_sum_err = abs(x[3] + x[4] + x[5] - 1.0)
        h_sum_err = abs(x[8] + x[9] + x[10] - 1.0)
        energy_err = abs(energy_fn(x))
        cal_surplus = calorie_fn(x)

        linear_ok = max(k_sum_err, l_sum_err, h_sum_err) < 1e-4
        energy_ok = energy_err < 1e-4
        calorie_ok = cal_surplus >= -1e-4
        obj_ok = np.isfinite(result.fun)
    except Exception:
        # if we can't even evaluate constraints, it's a failure
        return -1

    # if calorie constraint is badly violated, it's infeasible
    if cal_surplus < -1e-2:
        return -2

    # if all constraints are satisfied and objective is finite, accept it
    # regardless of what ipopt's status code says
    if linear_ok and energy_ok and calorie_ok and obj_ok:
        return 1

    # if ipopt explicitly says infeasible, trust that
    if status == 2:
        return -2

    # if ipopt says max iterations, report that
    if status == -1:
        return 0

    # calorie constraint is close to violated — likely infeasible region
    if not calorie_ok:
        return -2

    # everything else is a genuine failure
    return -1
