"""
runs a small grid of (tau, pop) combinations and saves results to csv
for easy comparison with matlab output. uses the same wrapper logic
so the fixed-point iteration, convergence checks, etc. are identical.
"""

import numpy as np
import csv
import warnings
from datetime import datetime
from ecc_parameters import get_parameters
from ecc_optimization import ecc_optimization

warnings.filterwarnings('ignore')

# ---- pick how many grid points you want ----
# these are the first few values from the full matlab grid:
#   tau = 1:-0.01:0  -->  [1.00, 0.99, 0.98, ...]
#   pop = 1:10:1201  -->  [1, 11, 21, 31, ...]
# change these to run more or fewer combinations
tau_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
pop_values = list(range(1, 1002, 50))


# ---- setup (same as ecc_opt_wrapper.py) ----
parameters = get_parameters()
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

# analytic solution for initial guess (same formulas as matlab wrapper)
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

# ---- run the grid and collect results ----
max_iter = 200
damping = 0.6

# column names for the csv
header = [
    'tau', 'pop', 'utility', 'exitflag', 'converged_iter',
    'k_y', 'k_f', 'k_n', 'l_y', 'l_f', 'l_n',
    'n_y', 'n_f', 'h_y', 'h_f', 'h_n', 'kmax', 'nmax'
]

rows = []
total = len(tau_values) * len(pop_values)
count = 0

print(f"running {len(tau_values)} x {len(pop_values)} = {total} combinations...\n")

for tech in tau_values:
    for population in pop_values:
        count += 1
        L = lfpr * population

        # initial kmax and nmax guess (same as matlab)
        ece = tech * (0.868 - 0.22) + 0.22
        nmax = (ece * PAR * sols_analytic[10] * (H / L)
                * sols_analytic[2]**eta0 * sols_analytic[5]**eta1)
        Omega = (A * sols_analytic[0]**gamma0 * sols_analytic[3]**gamma1
                 * (sols_analytic[6] * nmax)**gamma2
                 * (sols_analytic[8] * (H / L))**gamma3)
        kmax = (s / d * Omega) ** (1 / (1 - gamma0))
        parameters['nmax'] = nmax
        parameters['kmax'] = kmax

        # tolerance (same as matlab)
        if L <= 1:
            tolerance = 1e-4
        else:
            scale_factor = max(1, population / 25)
            tolerance = 1e-5 * scale_factor

        # fixed-point iteration (same as matlab)
        result_util = np.nan
        result_sol = np.full(11, np.nan)
        result_flag = -99
        result_iter = -1
        result_kmax = np.nan
        result_nmax = np.nan

        for iter_num in range(max_iter):
            try:
                soltemp, utiltemp, exitflag = ecc_optimization(population, tech, parameters)
            except Exception:
                result_flag = -1
                break

            if exitflag == -2:
                result_flag = -2
                break
            elif exitflag <= 0:
                result_flag = exitflag
                break

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
                result_util = -utiltemp  # flip sign back to positive utility
                result_sol = soltemp
                result_flag = 1
                result_iter = iter_num + 1
                result_kmax = kmax
                result_nmax = nmax
                break

            parameters['kmax'] = damping * kmax_implied + (1 - damping) * parameters['kmax']
            parameters['nmax'] = damping * nmax_implied + (1 - damping) * parameters['nmax']
            kmax = parameters['kmax']
            nmax = parameters['nmax']

            if iter_num == max_iter - 1:
                result_flag = 0
                result_iter = max_iter

        # build the row
        row = [
            f"{tech:.2f}",
            f"{population}",
            f"{result_util:.10e}" if np.isfinite(result_util) else str(result_util),
            str(result_flag),
            str(result_iter),
        ]
        for val in result_sol:
            row.append(f"{val:.10e}" if np.isfinite(val) else str(val))
        row.append(f"{result_kmax:.10e}" if np.isfinite(result_kmax) else str(result_kmax))
        row.append(f"{result_nmax:.10e}" if np.isfinite(result_nmax) else str(result_nmax))
        rows.append(row)

        # print progress
        status = "ok" if result_flag == 1 else f"flag={result_flag}"
        util_str = f"{result_util:.6f}" if np.isfinite(result_util) else str(result_util)
        print(f"  [{count}/{total}] tau={tech:.2f} pop={population:>4}  util={util_str:>12}  {status}")

# ---- save to csv ----
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_filename = f"ECC_comparison_{timestamp}.csv"

with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"\nresults saved to {csv_filename}")
print(f"{len(rows)} rows written")
