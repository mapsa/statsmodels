# -*- coding: utf-8 -*-
"""

Created on Wed May 09 22:17:39 2012

Author: Josef Perktold
"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from statsmodels.graphics.regressionplots import (utils,
                               plot_ccpr_ax, plot_partregress_ax)

def plot_regress_exog(res, exog_idx, exog_name='', fig=None):
    """Plot regression results against one regressor.

    This plots four graphs in a 2 by 2 figure: 'endog versus exog',
    'residuals versus exog', 'fitted versus exog' and
    'fitted plus residual versus exog'

    Parameters
    ----------
    res : result instance
        result instance with resid, model.endog and model.exog as attributes
    exog_idx : int
        index of regressor in exog matrix
    fig : Matplotlib figure instance, optional
        If given, this figure is simply returned.  Otherwise a new figure is
        created.

    Returns
    -------
    fig : matplotlib figure instance

    Notes
    -----
    This is currently very simple, no options or varnames yet.

    """

    fig = utils.create_mpl_fig(fig)

    if exog_name == '':
        exog_name = 'variable %d' % exog_idx

    #maybe add option for wendog, wexog
    #y = res.endog
    x1 = res.model.exog[:,exog_idx]
    from statsmodels.sandbox.regression.predstd import wls_prediction_std
    prstd, iv_l, iv_u = wls_prediction_std(res)

    ax = fig.add_subplot(2,2,1)
    #namestr = ' for %s' % self.name if self.name else ''
    ax.plot(x1, res.model.endog, 'o', color='b', label='endog', alpha=0.9)
    ax.plot(x1, res.fittedvalues, 'D', color='r', label='fitted', alpha=0.5)
    ax.vlines(x1, iv_l, iv_u, linewidth=1, color='k', alpha=0.7)
    ax.set_title('endog and fitted versus exog', fontsize='small')# + namestr)
    ax.legend(loc='lower right')


    ax = fig.add_subplot(2,2,3)
    #namestr = ' for %s' % self.name if self.name else ''
    ax.plot(x1, res.resid, 'o')
    ax.axhline(y=0)
    ax.set_title('residuals versus exog', fontsize='small')# + namestr)

    ax = fig.add_subplot(2,2,2)
    exog_noti = np.ones(res.model.exog.shape[1], bool)
    exog_noti[exog_idx] = False
    exog_others = res.model.exog[:, exog_noti]
    plot_partregress_ax(res.model.endog, x1, exog_others, varname='',
                        title_fontsize='small', ax=ax)
    x0 = ax.get_lines()[0]._x
    ax.set_xlim(x0.min(), x0.max())

#    ax = fig.add_subplot(2,3,4)
#    #namestr = ' for %s' % self.name if self.name else ''
#    ax.plot(x1, res.fittedvalues, 'o')
#    ax.set_title('Fitted versus exog', fontsize='small')# + namestr)

#    ax = fig.add_subplot(2,3,4)
#    #namestr = ' for %s' % self.name if self.name else ''
#    ax.plot(x1, res.fittedvalues + res.resid, 'o')
#    ax.set_title('Fitted plus residuals versus exog', fontsize='small')# + namestr)

    ax = fig.add_subplot(2,2,4)
    plot_ccpr_ax(res, exog_idx=exog_idx, ax=ax)
    ax.title.set_fontsize('small')


    fig.suptitle('Regression Plots for %s' % exog_name)

    return fig
