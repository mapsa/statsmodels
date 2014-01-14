# -*- coding: utf-8 -*-
"""

Created on Sun Jan 12 23:13:43 2014

Author: Josef Perktold
"""

import numpy as np
import statsmodels.stats.gof_precedence as prec

from numpy.testing import assert_allclose


x2 = np.asarray('0.00 0.18 0.55 0.66 0.71 1.30 1.63 2.17 2.75 10.60'.split(), float)
x3 = np.asarray('0.49 0.64 0.82 0.93 1.08 1.99 2.06 2.15 2.57 4.75'.split(), float)
x6 = np.asarray('1.34 1.49 1.56 2.10 2.12 3.83 3.97 5.13 7.21 8.71'.split(), float)

z0 = np.array([2, 3, 4, 6, 7, 8, 9, 11])
z1 = np.array([1, 5, 10, 12])

r_cut, n1, n2 = 7, 10, 10
res = prec.precedence(x3, x2, r_cut, n1, n2, kind='min', pvalue_method='auto')
print res.summary()
respv = prec.pvalue_normal(res.w_min, r_cut, n1, n2)
pvalues_normal = [respv.pvalue]
print
print respv.summary()
respv = prec.pvalue_normal(res.w_max, r_cut, n1, n2, kind='max')
pvalues_normal.append(respv.pvalue)
print
print respv.summary()
respv = prec.pvalue_normal(res.w_mean, r_cut, n1, n2, kind='mean')
pvalues_normal.append(respv.pvalue)
print
print respv.summary()

pd = prec.PrecedenceDistribution(n1, r_cut, n2)

v = pd.moments_sum
print '\n v:', v

wmin = pd.precedence_properties(sk=True)
print '\n wmin:', wmin
wmax = pd.precedence_properties(kind='max', sk=True)
print '\n wmax:', wmax
wmean = pd.precedence_properties(kind='mean', sk=True)
print '\n wmean:', wmean

pvalues_exact =(pd.pvalue(110), pd.pvalue(125, kind='max'), pd.pvalue(117.5, kind='mean'))
print '\n p-values:', pvalues_exact

m = prec.get_count(x3, x2)
print '\nwmin test: ', pd.precedence_test(m[:7])
print '\nwmax test: ', pd.precedence_test(m[:7], kind='max')
print '\nwmean test:', pd.precedence_test(m[:7], kind='mean')

res_exact = prec.precedence(x3, x2, r_cut, n1, n2, kind='min', pvalue_method='exact')
print res_exact.summary()

ppd = prec.PrecedencePermutationDistribution(n1, r_cut, n2)
ppd.run_permutation(n_rep=10000)
print '\n p-values:', (ppd.pvalue(110), ppd.pvalue(125, kind='max'),
                       ppd.pvalue(117.5, kind='mean'))

ppd.run_permutation(n_rep=10000)
p_values_perm = (ppd.pvalue(110), ppd.pvalue(125, kind='max'),
                 ppd.pvalue(117.5, kind='mean'))
print ppd._cache['bin_sum'].shape, 20000

w_mom_min = prec.wmoments_from_mmoments(v, n1, r_cut, n2, kind='min')
print '\nwmin mom: ', w_mom_min
w_mom_max = prec.wmoments_from_mmoments(v, n1, r_cut, n2, kind='max')
print '\nwmax mom: ', w_mom_max
w_mom_mean = prec.wmoments_from_mmoments(v, n1, r_cut, n2, kind='mean')
print '\nwmean mom: ', w_mom_mean

vv = prec.mmoments(n1, n2, r_cut)
print v
print vv
assert_allclose(vv, v, rtol=13)

print pvalues_exact
print p_values_perm
print np.squeeze(np.asarray(pvalues_normal))
