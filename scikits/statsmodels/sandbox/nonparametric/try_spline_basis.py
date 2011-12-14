# -*- coding: utf-8 -*-
"""Learning what a spline basis looks like - an exercise

Created on Thu Dec 08 10:36:19 2011

Author: Josef Perktold


splines are currently defined on integers with fixed interval length
simplest case
reference used: http://ego.psych.mcgill.ca/misc/fda/ex-basis-c2.html

http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-basis.html :
quote: "Basis function Ni,p(u) is non-zero on [ui, ui+p+1). Or, equivalently, Ni,p(u) is non-zero on p+1 knot spans [ui, ui+1), [ui+1, ui+2), ..., [ui+p, ui+p+1)."

changes:
added non-uniform b-splines and cache

no checking of boundary conditions yet
no boundary behavior is imposed, splines are extended on both sides.

I'm using splines just as a function basis. No special features of splines
are used, except that the function base is B-Splines.

missing:
there should be something on derivatives, I assume. DONE
ref on derivatives
http://www.cs.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-derv.html
Jana Prochazkova DERIVATIVE OF B-SPLINE FUNCTION, 25. KONFERENCE O GEOMETRII A POˇC´ITAˇCOV´E GRAFICE
http://mat.fsv.cvut.cz/gcg/sbornik/prochazkova.pdf

multiple knots, several knots at the same location - not tried yet, nee 0/0 = 0

TODO
the rest is mostly cleanup



"""

import numpy as np
import matplotlib.pyplot as plt

def basis0(t, j):
    return (np.abs(t-j-0.5) <= 0.5).astype(float)

maxj = 20 #global
def basis(t, j, m):
    '''spline basis function evaluated at points t with integer knot at j

    '''

#    print 'j,m', j,m, maxj
    mm1 = np.maximum(m-1., 1.)
    if m == 0:
        bjmt = basis0(t, j)
    else:
        if j == -1:
            print 'at j=-1'
            bjmt = 1./mm1 * ((m+j-t) * basis(t, j+1, m-1))
        elif j == maxj:
            print 'at maxj'
            bjmt = 1./mm1 * ((t-j) * basis(t, j, m-1))
        else:
            bjmt = 1./mm1 * ((t-j) * basis(t, j, m-1) +
                                (1+m+j-t) * (basis(t, j+1, m-1))) #- (t==j)
                                                         #double counting
                                                         #remove below -1e-15


    return bjmt * (np.abs((t-j)*1./np.maximum(2*m,1) - 0.5 -1e-15) <= 0.50000)

def get_bspline_basis(t, js, order, knots):
    #nested function to use knots and cache as global to recursion

    cache = {}
    maxj = len(knots)

    def basis_non_uniform(t, j, m):
        '''spline basis function evaluated at points t with integer knot at j

        '''
        if (j,m) in cache:
            #print 'hitting cache with', j, m
            return cache[(j,m)]

    #    print 'j,m', j,m, maxj
        if m == 0:
            bjmt = ((knots[j] <= t) & (t < knots[j+1])).astype(float)
        else:
            if j == -1:
                print 'at j=-1'  #this never happens ?
                coef_upp = np.maximum(knots[j+m+1] - t, 0) / (knots[j+m+1] - knots[j+1])
                bjmt = coef_upp * basis_non_uniform(t, j+1, m-1)
            elif j == maxj:
                print 'at maxj'
                coef_low = np.maximum(t - knots[j], 0) / (knots[j+m] - knots[j])
                bjmt = coef_low * basis_non_uniform(t, j, m-1)
            else:
                #Note: from reference, if there is 0/0, then 0/0=0
                #if (knots[j+m] - knots[j]) > 0: TODO
                coef_low = np.maximum(t - knots[j], 0) / (knots[j+m] - knots[j])
                coef_upp = np.maximum(knots[j+m+1] - t, 0) / (knots[j+m+1] - knots[j+1])
                bjmt = (coef_low * basis_non_uniform(t, j, m-1) +
                       coef_upp * basis_non_uniform(t, j+1, m-1))

        #bjmt = bjmt * ((knots[j] <= t) & (t < knots[j+m+1]))
        #not quite sure why I don't need the support indicator?
        #I only use maximum above and don't constrain upper bound
        cache[(j,m)] = bjmt
        #print 'number of functions evaluated', len(cache)
        return bjmt

    basis_funcs = np.array([basis_non_uniform(t, j, order) for j in js])
    m = order
    derivatives = np.array([m * (
                   1./(knots[j+m] - knots[j]) * basis_non_uniform(t, j, m-1) -
                   1./(knots[j+m+1] - knots[j+1]) * basis_non_uniform(t, j+1, m-1))
                    for j in js])

    #cache will go out of scope on return
    return basis_funcs, derivatives




# example using regression

nobs = 500
sige = 0.3
t = np.linspace(0,6, nobs)
x_dgp = np.column_stack((np.arange(nobs)/nobs,
                         np.sin(t**2), #1.75),
                         np.exp(t)/np.exp(t.max())))
beta = np.ones((x_dgp.shape[1], 1))
beta[-1] = 3 #exp trend
y_true = np.dot(x_dgp, beta)
np.random.seed(98765)
y = y_true + sige * np.random.normal(size=(nobs,1))

y[nobs//2:nobs//2+25:3] = 3 #+= 5 * np.abs(y[nobs//2:nobs//2+25:5])

order = 3 #or is it order+1
knots = range(0, int(t.max()))  #currently integer only

#construct regressor: constant and spline basis
exog = np.column_stack([np.arange(nobs)] + [basis(t, j, order) for j in knots])


from scikits.statsmodels.regression.linear_model import OLS
res = OLS(y, exog).fit()
print 'estimated parameters and tvalues'
print res.params
print res.tvalues

plt.figure()
plt.plot(t, y, 'o')
plt.plot(t, y_true, 'k-', label='true')
plt.plot(t, res.fittedvalues, label='fitted')
plt.legend()
plt.title('Least squares fit to spline basis (DIY)')


# some plots for basis functions

t_all = np.linspace(-1, 11, 501)
plt.figure()
plt.title('Spline Basis Functions')
for j in range(8):
    #plt.plot(t_all, basis0(t_all, j), '-o')
    plt.plot(t_all, basis(t_all, j, 2), '-o')

plt.figure()
knots = 0.5 * np.arange(maxj)**2
j = 3
m = 3
t_all = np.linspace(-1, 51, 501)

js = range(8)
base0, deriv0  = get_bspline_basis(t_all, js, m, knots)  #spline functions in rows
for j in range(8):
    #plt.plot(t_all, basis_non_uniform(t_all, j, m, knots), '-')
    plt.plot(t_all, base0[j], '-')
    plt.title('Non-Uniform B-Spline Basis Functions')

#check derivative
for i in range(8):
    print np.max(np.abs(np.gradient(base0[i,:], (t_all[1]-t_all[0])) - deriv0[i, :]))

#plt.figure()
#plt.plot(t_all[:-1]+(t_all[1]-t_all[0])/2, np.diff(base0[4,:]) / np.diff(t_all))
#plt.plot(t_all, deriv0[4])
#plt.title('derivative of B-spline - checking')

#plt.figure()
#plt.plot(t_all, basis0(t_all, 0), '-o')
#plt.plot(t_all, basis0(t_all, 1), '-o')
#plt.plot(t_all, basis(t_all, 0, 1))


#--------------- new example with equal spaced knots at float intervals

#this works now, still need to check the boundary conditions, enough knots
#on each end? or too many?
#I don't understand, in the front they are all significant, but not at the end

order = 3
n_knots = 4
res_all = []
bics = []
for n_knots in range(1,25):
    diff = (t.max() - t.min()) / float(n_knots)
    #equal spaced
    #extended knots
    #knots = np.linspace(t.min()-order*diff, t.max()+(order+1)*diff, n_knots + 2*order+1+1)
    #js = range(order, order+n_knots-1)
    uniform = False
    if uniform:
        knots = np.linspace(t.min()-(order)*diff, t.max()+(order+1+3)*diff, n_knots + 2*order+1+1+3) #+3 for safety
    else:
        #integer problem in the middle when it's not an even number ???
        #see np.diff(knots)  SOLVED now
        knots = np.linspace(t.min()-(order)*diff*2, t.mean(), (n_knots + 2*order+1+1)//2)
        #knots2 = np.linspace(t.mean()+diff*0.5, t.max()+(order+1+3)*diff*0.5, (n_knots + 2*order+1+1+3)//2) #+3 for safety
        remain = n_knots - (len(knots) - order - 1)
        remain = max(remain, 1)
        diff2 = (t.max() - t.mean()) / float(remain)
        #knots2 = np.linspace(t.mean(), t.max()+(order+1+1)*diff*0.5, (n_knots + 2*order+1+1)//2+1+1)[1:]
        knots2 = np.linspace(t.mean(), t.max()+(order+1+1)*diff2, (remain + order+1+1)+1)[1:]
        knots = np.concatenate((knots, knots2))
    js = range(0, order+n_knots)
    base, _ = get_bspline_basis(t, js, order, knots)

    #construct regressor: constant and spline basis
    exog = np.column_stack((np.arange(nobs), base.T))


    from scikits.statsmodels.regression.linear_model import OLS
    res = OLS(y, exog).fit()
    print 'estimated parameters and tvalues'
    print res.params
    print res.tvalues
    print 'BIC', res.bic, 'n_knots=', n_knots
    bics.append(res.bic)
    res_all.append(res)

best_idx = np.argmin(bics)
n_knots = range(1,25)[best_idx]
#mistake setting n_knots but not reestimating
res = res_all[best_idx]
#res = res_all[0]

import scikits.statsmodels.api as sm
res_rlm = sm.RLM(res.model.endog, res.model.exog).fit()

plt.figure()
plt.plot(t, y, 'k.')
plt.plot(t, y_true, 'k-', lw=2, label='true')
plt.plot(t, res.fittedvalues, 'b', lw=2, label='fitted OLS')
plt.plot(t, res_rlm.fittedvalues, 'r', lw=2, label='fitted RLM')
plt.legend()
plt.title('Least squares fit to spline basis (with outliers) %d knots' % n_knots)




plt.show()
