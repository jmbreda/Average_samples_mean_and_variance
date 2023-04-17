def log_likelihood(t, mean, variances):
    """ Calculate likelihood """
    alpha = 1. / (variances + t * t)
    mSqrt = np.sqrt(np.sum(alpha))
    m0 = np.sum(alpha)
    m1 = np.sum(alpha * mean)
    m2 = np.sum(alpha * np.power(mean, 2))
    mu = m1 / m0
    return (0.5 * (m0 * mu * mu - 2 * mu * m1 + m2) - np.log(mSqrt))

def find_tau(mean, variances):
    """ finds tau """
    (tau, fval, ierr, numfunc) = so.fminbound(log_likelihood, 0, 10000,args=(E, variances),full_output=1, xtol=1.e-06)
    if ierr:
        sys.exit("log_E has not converged after %d iterations.\n"% numfunc)
    alpha = 1. / (variances + np.power(tau, 2))
    m0 = np.sum(alpha)
    m1 = np.sum(alpha * mean)
    mu = m1 / m0
    new_sig = np.sqrt(1 / m0)
    return (tau, mu, new_sig)
