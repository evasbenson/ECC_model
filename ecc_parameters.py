"""
translated from ECC_parameters.m
contains all the model parameters as a dictionary instead of a matlab struct
"""


def get_parameters():
    """returns a dictionary of all model parameters, same values as the matlab file"""
    params = {
        # labor and demographics
        'lfpr': 0.75 * 0.61,  # 75% working age x 61% lfpr among working age pop
        'H': 4.9130,           # billion hectares of land developed or in agriculture

        # utility function parameters
        'mu0': 0.985,
        'mu1': 0.015,

        # economic parameters
        's': 0.26,      # savings rate
        'd': 0.0465,    # depreciation rate
        'A': 4.8484,    # calibrated TFP

        # aggregate production function parameters (gamma)
        'gamma0': 0.285,  # capital share
        'gamma1': 0.625,  # labor share
        'gamma2': 0.04,   # energy share
        'gamma3': 0.05,   # land share

        # food production function parameters (alpha)
        'alpha0': 0.335,  # capital share in kcal production
        'alpha1': 0.625,  # labor share in kcal production
        'alpha2': 0.04,   # energy share in kcal production

        # energy production function parameters (eta)
        'eta0': 0.375,  # capital share in energy production
        'eta1': 0.625,  # labor share in energy production

        # other parameters
        'PAR': 2164.5,        # photosynthetically active radiation (MJ per sq meter per year)
        'kcalmin': 1500 * 365, # minimum kcal per capita per year (1500 per day)
    }

    return params
