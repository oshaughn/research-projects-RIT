# Coordinate transformation tools
# https://git.ligo.org/soichiro.morisaki/implement_mu1_mu2_into_bilby/-/blob/master/tools.py

# Some truncations to avoid introducing external dependencies




import numpy as np


U = np.array(
    [[0.97437198,  0.20868103,  0.08397302],
     [-0.22132704,  0.82273827,  0.52356096],
     [0.04016942, -0.52872863,  0.84783987]]
)
fref = 200.
MsunToTime = 4.92659 * 10.0**(-6.0)


def m1m2ToMc(m1, m2):
    return (m1 * m2)**(3. / 5.) / ((m1 + m2)**(1. / 5.))


def qToeta(q):
    """q=m2/m1 to eta=m1*m2/(m1+m2)**2"""
    return q / ((1 + q)**2.0)


def etaToq(eta):
    """eta=m1*m2/(m1+m2)**2 to q=m2/m1"""
    tmp = 1. - 1. / (2. * eta)
    return - tmp - np.sqrt(tmp**2. - 1.)


def McqToMtot(Mc, q):
    return Mc * qToeta(q)**(-3./5.)


def McqTom1m2(Mc, q):
    Mtot = McqToMtot(Mc, q)
    return Mtot / (1 + q), Mtot * q / (1 + q)

def m1m2(Mc,eta):
    return McqTom1m2(Mc, etaToq(eta))


def psi0(Mc):
    """chirpmass to the 0PN phase coefficient"""
    return (3. / 4.) * (8. * np.pi * Mc * MsunToTime * fref)**(-5. / 3.)


def psi2(Mc, eta):
    """chirpmass and eta=m1*m2/(m1+m2)**2 to the 1PN phase coefficient"""
    return psi0(Mc) * (20. / 9.) * (743. / 336. + 11. * eta / 4.) * \
        eta**(-2. / 5.) * (np.pi * Mc * MsunToTime * fref)**(2. / 3.)


def qa1za2zTobeta(q, a1z, a2z):
    return ((113. / 12. + 25. * q / 4.) * a1z + \
        q**2. * (113. / 12. + 25. / (4. * q)) * a2z) / ((1. + q)**2.)


def psi3(Mc, q, a1z, a2z):
    beta = qa1za2zTobeta(q, a1z, a2z)
    eta = qToeta(q)
    return psi0(Mc) * (4. * beta - 16. * np.pi) \
        * eta**(-3. / 5.) * np.pi * Mc * MsunToTime * fref


def _cancel_psi3(Mc, eta):
    return (U[0, 0] - U[1, 0] * U[0, 2] / U[1, 2]) * psi0(Mc) + \
        (U[0, 1] - U[1, 1] * U[0, 2] / U[1, 2]) * psi2(Mc, eta)


def _mu1mu2etaToMc(mu1, mu2, eta):
    """Convert mu1, mu2, eta=m1*m2/(m1+m2)**2 into chirpmass using bisection
    search. This assumes the inputs are floats"""
    psi3 = mu1 - (U[0, 2] / U[1, 2]) * mu2;
    mcmin = mcmax = (128. * mu1 / 3.)**(-3. / 5.) / (np.pi * fref * MsunToTime)
    while _cancel_psi3(mcmax, eta) >= psi3:
        mcmax = mcmax * 2.
    while _cancel_psi3(mcmin, eta) < psi3:
        mcmin = mcmin * 0.5
    mcmid = (mcmin + mcmax) / 2.
    while ((mcmax - mcmin) / mcmin) > 10.**(-6.):
        if _cancel_psi3(mcmid, eta) > psi3:
            mcmin = mcmid
        else:
            mcmax = mcmid
        mcmid = (mcmin + mcmax) / 2.
    return mcmid


def mu1mu2etaToMc(mu1, mu2, eta):
    """Convert mu1, mu2, eta=m1*m2/(m1+m2)**2 into chirpmass using bisection
    search."""
    if type(mu1) is np.ndarray:
        return np.array([_mu1mu2etaToMc(_mu1, _mu2, _eta) for _mu1, _mu2, _eta in zip(mu1, mu2, eta)])
    else:
        return  _mu1mu2etaToMc(mu1, mu2, eta)


def mu2Mcetachi2Tochi1(mu2, Mc, eta, chi2):
    """mu2, chirpmass, eta=m1*m2/(m1+m2)**2, chi2 to chi1"""
    _psi0 = psi0(Mc)
    beta = 4. * np.pi + (mu2 - U[1, 0] * _psi0 - U[1, 1] * psi2(Mc, eta)) / (U[1, 2] * 4. * np.pi * Mc * MsunToTime * fref * eta**(-3. / 5.) * _psi0)
    q = etaToq(eta)
    return ((1. + q)**2. * beta - q**2. * (113. / 12. + 25. / (4. * q)) * chi2) / (113. / 12. + 25. * q / 4.)


def mu1mu2qchi2ToMcqchi1chi2(mu1, mu2, q, chi2):
    """mu1, mu2, q=m2/m1, chi2 to chirpmass, q, chi1, chi2"""
    eta = qToeta(q)
    Mc = mu1mu2etaToMc(mu1, mu2, eta)
    chi1 = mu2Mcetachi2Tochi1(mu2, Mc, eta, chi2)
    return (Mc, q, chi1, chi2)

# Vinaya's versions, for AMR
def transform_mu1mu2qs2z_m1m2s1zs2z(mu1, mu2, q, s2z):
    """mu1, mu2, q=m2/m1, z-component of secondary spin to component masses: m1, m2 and z-components of spins: s1z, s2z"""
    eta = qToeta(q)
    mc = mu1mu2etaToMc(mu1, mu2, eta)
    s1z = mu2Mcetachi2Tochi1(mu2, mc, eta, s2z)
#    m1, m2 = lalsimutils.m1m2(mc, eta)
    m1, m2 = McqTom1m2(mc, q)
    return m1, m2, s1z, s2z
def transform_m1m2s1zs2z_mu1mu2qs2z(m1, m2, s1z, s2z):
    """component masses: m1, m2 and z-components of spins: s1z, s2z to mu1, mu2, q=m2/m1, s2z"""
    mc, q = transform_m1m2_mcq(m1, m2)
    eta = qToeta(q)
    psi0 = mcTopsi0(mc)
    psi2 = mcetaTopsi2(mc, eta)
    psi3 = mcqs1zs2zTopsi3(mc, q, s1z, s2z)
    return (
        mu_coeffs[0, 0] * psi0 + mu_coeffs[0, 1] * psi2 + mu_coeffs[0, 2] * psi3,
        mu_coeffs[1, 0] * psi0 + mu_coeffs[1, 1] * psi2 + mu_coeffs[1, 2] * psi3,
        q,
        s2z
    )

def Mcqchi1chi2Tomu1mu2mu3(mc, q, chi1, chi2):
    """chirpmass, q, chi1, chi2 to mu1, mu2, mu3"""
    psis = np.array([psi0(mc), psi2(mc, qToeta(q)), psi3(mc, q, chi1, chi2)])
    return U.dot(psis)


def convert_Mcqchi1chi2_to_mu1mu2(parameters):
    """
    Convert Mc, q, chi1, chi2 to mu1, mu2 and add them to parameters.

    Parameters
    ----------
    parameters: dict

    Returns
    -------
    converted_parameters: dict
    """
    converted_parameters = parameters.copy()
    psis = np.array([
        psi0(parameters["chirp_mass"]),
        psi2(parameters["chirp_mass"], qToeta(parameters["mass_ratio"])),
        psi3(parameters["chirp_mass"], parameters["mass_ratio"], parameters["chi_1"], parameters["chi_2"])
    ])
    mu1, mu2, _ = np.dot(U, psis)
    converted_parameters["mu_1"] = mu1
    converted_parameters["mu_2"] = mu2
    return converted_parameters


def convert_m1m2chi1chi2_to_Mcqmu1mu2(parameters):
    """
    Convert m1, m2, chi1, chi2 to Mc, q, mu1, mu2 and add them to parameters.

    Parameters
    ----------
    parameters: dict

    Returns
    -------
    converted_parameters: dict
    """
    converted_parameters = parameters.copy()
    converted_parameters["chirp_mass"] = m1m2ToMc(parameters["mass_1"], parameters["mass_2"])
    converted_parameters["mass_ratio"] = parameters["mass_2"] / parameters["mass_1"]
    return convert_Mcqchi1chi2_to_mu1mu2(converted_parameters)


def convert_mu1mu2qchi2_to_Mcchi1(parameters):
    """
    Convert mu1, mu2, q, chi2 to Mc, chi1 and add them to parameters.

    Parameters
    ----------
    parameters: dict

    Returns
    -------
    converted_parameters: dict
    """
    converted_parameters = parameters.copy()

    Mc, _, chi_1, _ = mu1mu2qchi2ToMcqchi1chi2(
        parameters["mu_1"], parameters["mu_2"], parameters["mass_ratio"], parameters["chi_2"])
    converted_parameters["chirp_mass"] = Mc
    converted_parameters["chi_1"] = chi_1

    return converted_parameters




def m1m2chi1chi2Tomu1mu2qchi2Jacobian(Mc, q):
    """Jacobian of the transformation from m1, m2, chi1, chi2 to mu1, mu2, q, chi2"""
    coef1 = U[0, 2] * U[1, 0] - U[0, 0] * U[1, 2]
    coef2 = U[0, 2] * U[1, 1] - U[0, 1] * U[1, 2]
    eta = qToeta(q)
    x = np.pi * MsunToTime * Mc * fref
    m1, _ = McqTom1m2(Mc, q)
    numerator = x * 5. * (113. + 75. * q) * \
        (252. * coef1 * q * eta**(-3. / 5.) + \
         coef2 * (743. + 2410. * q + 743. * q**2.) * x**(2. / 3.))
    denominator = m1**2. * 4128768. * q * (1. + q)**2. * x**(10. / 3.)
    return np.abs(numerator / denominator)
