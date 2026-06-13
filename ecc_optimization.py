"""
ecc_optimization.py

Translated from ECC_optimization_v5.m

Key changes from previous version:
- Food and energy production use tanh() saturation
- Capital K passed in directly
- Energy adding-up: n_y_frac + n_f_frac = 1
- A parameter now read from parameters dict (default 1.0)
- x0 hardcoded as in MATLAB
"""

import numpy as np
from cyipopt import minimize_ipopt


def ecc_optimization(population, technology, K, parameters):
    """
    Solves the constrained utility maximization for a given population,
    technology level, and capital stock K.

    Decision variables x (11 elements):
      x[0]  = K_y / K      — capital share for aggregate good
      x[1]  = K_f / K      — capital share for food
      x[2]  = K_n / K      — capital share for energy
      x[3]  = L_y / L      — labor share for aggregate good
      x[4]  = L_f / L      — labor share for food
      x[5]  = L_n / L      — labor share for energy
      x[6]  = n_y_frac     — fraction of energy N to aggregate good
      x[7]  = n_f_frac     — fraction of energy N to food
      x[8]  = H_y / H      — land share for aggregate good
      x[9]  = H_f / H      — land share for food
      x[10] = H_n / H      — land share for energy

    Parameters
    ----------
    population : float — population in billions
    technology : float — tau in [0, 1]
    K : float — current capital stock
    parameters : dict — from ecc_parameters.get_parameters()

    Returns
    -------
    equilibrium : array (11,) — optimal allocation
    negutility : float — negative utility at optimum
    Y : float — aggregate production at optimum
    exitflag : int — 1=success, -2=infeasible, 0=max iter, -1=failure
    """
    pop  = population
    tau  = technology
    lfpr = parameters['lfpr']
    H    = parameters['H']
    L    = lfpr * pop

    mu_c    = parameters['mu_c']
    mu_f    = parameters['mu_f']
    kcalmin = parameters['kcalmin']
    s       = parameters['s']

    # A scales aggregate production — read from params, default 1.0
    A = parameters.get('A', 1.0)

    gamma_k = parameters['gamma_k']
    gamma_l = parameters['gamma_l']
    gamma_n = parameters['gamma_n']
    gamma_h = parameters['gamma_h']

    PAR      = parameters['PAR']          # global PAR — for solar energy
    PAR_ag   = parameters.get('PAR_ag', parameters['PAR'])  # agricultural PAR — for food
    PAR_kcal = 2390060 * PAR_ag 
    alpha_k  = parameters['alpha_k']
    alpha_l  = parameters['alpha_l']
    alpha_n  = parameters['alpha_n']
    c1       = parameters['c1']

    eta_k = parameters['eta_k']
    eta_l = parameters['eta_l']
    c2    = parameters['c2']
    ece   = tau * (0.868 - 0.22) + 0.22

    conversion = tau * (0.123 - 0.123 / 2.5) + 0.123 / 2.5
    harvest    = 1.0 - (tau * (0.0 - 0.5) + 0.5)
    g          = PAR_kcal * harvest * conversion

    def objective(x):
        K_y      = x[0] * K
        K_f      = x[1] * K
        K_n      = x[2] * K
        L_y      = x[3] * L
        L_f      = x[4] * L
        L_n      = x[5] * L
        n_y_frac = x[6]
        n_f_frac = x[7]
        H_y      = x[8]  * H
        H_f      = x[9]  * H
        H_n      = x[10] * H

        N   = ece * PAR * H_n * np.tanh(c2 * K_n**eta_k * L_n**eta_l / H_n)
        N_y = N * n_y_frac
        N_f = N * n_f_frac

        Y    = A * K_y**gamma_k * L_y**gamma_l * N_y**gamma_n * H_y**gamma_h
        C    = (1 - s) * Y
        c_pc = C / pop

        F    = g * H_f * np.tanh(c1 * K_f**alpha_k * L_f**alpha_l * N_f**alpha_n / H_f)
        f_pc = F / pop

        u = (c_pc ** mu_c) * (f_pc ** mu_f)
        return -u

    def get_Y(x):
        K_y  = x[0] * K
        L_y  = x[3] * L
        H_y  = x[8]  * H
        H_n  = x[10] * H
        K_n  = x[2]  * K
        L_n  = x[5]  * L
        N    = ece * PAR * H_n * np.tanh(c2 * K_n**eta_k * L_n**eta_l / H_n)
        N_y  = N * x[6]
        return A * K_y**gamma_k * L_y**gamma_l * N_y**gamma_n * H_y**gamma_h

    def calorie_surplus(x):
        K_f      = x[1] * K
        L_f      = x[4] * L
        n_f_frac = x[7]
        H_f      = x[9]  * H
        H_n      = x[10] * H
        K_n      = x[2]  * K
        L_n      = x[5]  * L
        N        = ece * PAR * H_n * np.tanh(c2 * K_n**eta_k * L_n**eta_l / H_n)
        N_f      = N * n_f_frac
        F        = g * H_f * np.tanh(c1 * K_f**alpha_k * L_f**alpha_l * N_f**alpha_n / H_f)
        f_pc     = F / pop
        return f_pc - kcalmin

    constraints = [
        {'type': 'eq',   'fun': lambda x: x[0] + x[1] + x[2] - 1.0},
        {'type': 'eq',   'fun': lambda x: x[3] + x[4] + x[5] - 1.0},
        {'type': 'eq',   'fun': lambda x: x[6] + x[7] - 1.0},
        {'type': 'eq',   'fun': lambda x: x[8] + x[9] + x[10] - 1.0},
        {'type': 'ineq', 'fun': calorie_surplus},
    ]

    bounds = [(0.0001, 0.9999)] * 11

    x0 = np.array([0.1, 0.2, 0.7, 0.1, 0.2, 0.7, 0.3, 0.7, 0.1, 0.2, 0.7])

    options = {
        'print_level': 0, 'sb': 'yes',
        'max_iter': 3000,
        'tol': 1e-8,
        'constr_viol_tol': 1e-8,
        'acceptable_tol': 1e-7,
        'acceptable_constr_viol_tol': 1e-7,
        'nlp_scaling_method': 'gradient-based',
        'mu_strategy': 'adaptive',
    }

    try:
        result   = minimize_ipopt(objective, x0, bounds=bounds,
                                  constraints=constraints, options=options)
        exitflag = _map_exit_flag(result, calorie_surplus)
        Y_val    = get_Y(result.x)
        return result.x, result.fun, Y_val, exitflag
    except Exception:
        return x0, objective(x0), get_Y(x0), -1


def _map_exit_flag(result, calorie_fn):
    x      = result.x
    status = getattr(result, 'status', -100)
    try:
        k_err  = abs(x[0] + x[1] + x[2] - 1.0)
        l_err  = abs(x[3] + x[4] + x[5] - 1.0)
        e_err  = abs(x[6] + x[7] - 1.0)
        h_err  = abs(x[8] + x[9] + x[10] - 1.0)
        cal    = calorie_fn(x)
        lin_ok = max(k_err, l_err, e_err, h_err) < 1e-4
        cal_ok = cal >= -1e-4
        obj_ok = np.isfinite(result.fun)
    except Exception:
        return -1

    if cal < -1e-2:
        return -2
    if lin_ok and cal_ok and obj_ok:
        return 1
    if status == 2:
        return -2
    if status == -1:
        return 0
    if not cal_ok:
        return -2
    return -1
