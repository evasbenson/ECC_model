from ecc_parameters import get_parameters
from ecc_optimization import ecc_optimization
import numpy as np

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

# --- set the single point to test here ---
tau = 0.5
pop = 100

L = params['lfpr'] * pop
ece = tau * (0.868 - 0.22) + 0.22
nmax = ece * params['PAR'] * sols[10] * (params['H']/L) * sols[2]**eta0 * sols[5]**eta1
Omega = params['A'] * sols[0]**gamma0 * sols[3]**gamma1 * (sols[6]*nmax)**gamma2 * (sols[8]*(params['H']/L))**gamma3
kmax = (params['s']/params['d'] * Omega)**(1/(1-gamma0))

params['nmax'] = nmax
params['kmax'] = kmax

sol, util, flag = ecc_optimization(pop, tau, params)
print(f"exitflag: {flag}")
print(f"utility:  {-util:.6f}")
print(f"solution: {np.round(sol, 4)}")