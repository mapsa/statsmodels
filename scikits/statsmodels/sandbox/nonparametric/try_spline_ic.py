# -*- coding: utf-8 -*-
"""choosing a smoothing spline by minimizing BIC (Bayesian Information
Criterium)

Created on Tue Dec 13 14:20:13 2011

Author: Josef Perktold


TODO:
From the docs it looks like UnivariateSpline could update for changed s
instead of re-estimating from the beginning.

"""

import numpy as np
from scipy import optimize, interpolate

#from scikits.statsmodels.tools.eval_measures import bic_sigma #, aicc_sigma
#copied
def bic_sigma(sigma2, nobs, df_modelwc):
    return np.log(sigma2) + (np.log(nobs) * df_modelwc) / nobs


def fit_spline_ic(y, x, sfact=None):
    '''choose least-squares spline by minimizing BIC

    arguments are reversed as in statsmodels


    Notes
    -----
    It is possible that the optimization might converge to an interpolating
    spline (s=1e-17 or so). To avoid this, we try different starting values
    for the optimization, and choose the smoothing parameter s as the one that
    minimizes BIC among those that have s>1e-14.

    '''

    def fun(s):
        s = len(y) * (s)**2  #force positive
        sp = interpolate.UnivariateSpline(x, y, s=s)
        return bic_sigma(sp.get_residual(), len(x), len(sp.get_knots()) + 3)

    if sfact is None:
        res = [optimize.fmin(fun, sfact*y.std()) for sfact in np.linspace(0,0.75,5)]
        s = len(y) * np.array(res)**2
        s_max = np.max(s[s>1e-14])
    else:
        res = optimize.fmin(fun, sfact*y.std())
        s_max = len(y) * np.array(res)**2

    return interpolate.UnivariateSpline(x, y, s=s_max), s_max


if __name__ == '__main__':
    nobs = 200
    x = np.linspace(0, 6.*np.pi, nobs)
    A = 1 + x/x[-1] # Amplitude of sine function
    #y_true = A * np.sin(x**2) #that's a funny one, counter example
    y_true = A * np.sin((x/3)**2)
    np.random.seed(9875345)
    #y = y_true + 0.2 * A * np.random.randn(nobs) #heteroscedastic noise
    y = y_true + 0.2 * np.random.randn(nobs)
    import time
    t0 = time.time()
    sp, s_max = fit_spline_ic(y, x) #, 0.2)  #much faster with good guess
    yinterp = sp(x)
    t1 = time.time()

    #requires https://github.com/jjstickel/scikit-datasmooth/blob/master/scikits/datasmooth/regularsmooth.py
    #in same directory
    try:
        import regularsmooth as rs
        has_sksmooth = True
    except ImportError:
        has_sksmooth = False

    if has_sksmooth:
        t2 = time.time()
        r = rs.smooth_data(x,y)
        t3 = time.time()


    from scikits.statsmodels.nonparametric import lowess as lo
    t4 = time.time()
    actual_lowess = lo.lowess(y,x)
    t5 = time.time()
    print 'times', t1-t0, t5-t4,
    if has_sksmooth:
        print t3-t2

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y, 'bo',label='observed')
    ax.plot(x, y_true, 'k',label='y true')
    ax.plot(x, yinterp, 'r', label='smoothed scipy BIC')
    if has_sksmooth:
        ax.plot(x, r[0], 'g', label='smoothed datasmooth')
    ax.plot(x, actual_lowess[:,1], label='smoothed lowess')

    ax.legend(loc='lower left')
    #ax.set_title('Smoothing Spline (min BIC), s=%4.2f, n_k=%d' % (s_max, len(sp.get_knots())))
    ax.set_title('Smoothers (default settings), s=%4.2f, n_k=%d' % (s_max, len(sp.get_knots())))
    plt.show()
