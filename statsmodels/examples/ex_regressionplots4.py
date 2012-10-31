# -*- coding: utf-8 -*-
"""Examples for Regression Plots

Author: Josef Perktold

"""

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

import statsmodels.graphics.regressionplots as smrp

#example from tut.ols with changes
#fix a seed for these examples
np.random.seed(9876789)

# OLS non-linear curve but linear in parameters
# ---------------------------------------------

nsample = 100
sig = 1. #0.5
x1 = np.linspace(0, 20, nsample)
x2 = 5 + 3 * np.random.randn(nsample)
x3 = 5 * np.random.randn(nsample) + 0.1 * x1
X = np.c_[x1, x2, x3, np.sin(0.5*x1), (x2-5)**2, np.ones(nsample)]
#beta = [0.5, 0.5, 1, -0.04, 5.]
beta = [0.5, 0.5, 0.5, 1, -0.04, 5.]
y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)

#estimate only linear function, misspecified because of non-linear terms
exog0 = sm.add_constant(np.c_[x1, x2, x3], prepend=False)

res = sm.OLS(y, exog0).fit()

def add_lowess(fig, ax_idx=0, lines_idx=0, frac=0.2):
    '''add lowess line to a plot
    '''
    ax = fig.axes[ax_idx]
    y0 = ax.get_lines()[lines_idx]._y
    x0 = ax.get_lines()[lines_idx]._x
    lres = sm.nonparametric.lowess(y0, x0, frac=frac)
    ax.plot(lres[:,0], lres[:,1], 'r', lw=1.5)
    return fig

all_plots = False
if all_plots:
    fig1 = smrp.plot_fit(res, 0, y_true=None)
    fig2 = smrp.plot_fit(res, 1, y_true=None)

    fig3 = smrp.plot_partregress(y, exog0, exog_idx=[0,1])
    add_lowess(fig3, ax_idx=0, lines_idx=0)
    add_lowess(fig3, ax_idx=1, lines_idx=0)

    fig4 = smrp.plot_regress_exog(res, exog_idx=0)
    add_lowess(fig4, ax_idx=1, lines_idx=0)
    add_lowess(fig4, ax_idx=3, lines_idx=0)

    fig5 = smrp.plot_regress_exog(res, exog_idx=1)
    add_lowess(fig5, ax_idx=1, lines_idx=0)
    add_lowess(fig5, ax_idx=3, lines_idx=0)

    fig6 = smrp.plot_ccpr(res, exog_idx=[0])
    add_lowess(fig6, ax_idx=0, lines_idx=0)

    fig7 = smrp.plot_ccpr(res, exog_idx=[0,1])
    add_lowess(fig7, ax_idx=0, lines_idx=0)
    add_lowess(fig7, ax_idx=1, lines_idx=0)


    fig8 = smrp.plot_partregress(y, exog0, exog_idx=[0,1])
    #add lowess
    ax = fig8.axes[0]
    y0 = ax.get_lines()[0]._y
    x0 = ax.get_lines()[0]._x
    lres = sm.nonparametric.lowess(y0, x0, frac=0.2)
    ax.plot(lres[:,0], lres[:,1], 'r', lw=1.5)
    ax = fig8.axes[1]
    y0 = ax.get_lines()[0].get_ydata()
    x0 = ax.get_lines()[0].get_xdata()
    lres = sm.nonparametric.lowess(y0, x0, frac=0.2)
    ax.plot(lres[:,0], lres[:,1], 'r', lw=1.5)

#from regressionplots_2by3 import plot_regress_exog
from regressionplots_new import plot_regress_exog
fig9 = plot_regress_exog(res, exog_idx=0)
add_lowess(fig9, ax_idx=1, lines_idx=0)
add_lowess(fig9, ax_idx=2, lines_idx=0)
add_lowess(fig9, ax_idx=3, lines_idx=0)
#add_lowess(fig9, ax_idx=5, lines_idx=0)

fig10 = plot_regress_exog(res, exog_idx=1)
add_lowess(fig10, ax_idx=1, lines_idx=0)
add_lowess(fig10, ax_idx=2, lines_idx=0)
#add_lowess(fig10, ax_idx=4, lines_idx=0)
add_lowess(fig10, ax_idx=3, lines_idx=0)

fig11 = plot_regress_exog(res, exog_idx=2)
add_lowess(fig11, ax_idx=1, lines_idx=0)
add_lowess(fig11, ax_idx=2, lines_idx=0)
#add_lowess(fig10, ax_idx=4, lines_idx=0)
add_lowess(fig11, ax_idx=3, lines_idx=0)


#plt.show()
