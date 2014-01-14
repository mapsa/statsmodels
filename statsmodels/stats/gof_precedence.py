# -*- coding: utf-8 -*-
"""Wilcoxon-Type precedence test for equality of distribution against
one-sided alternative

Created on Sun Jan 12 22:32:33 2014

Author: Josef Perktold
License: BSD-3

"""

from __future__ import division
import numpy as np
from scipy import stats
from scipy.misc import comb

from statsmodels.tools.decorators import cache_readonly

class Holder(object):

    def __init__(self, **kwds):
        self.__dict__.update(kwds)
        self.kwds = kwds.keys()

    def summary(self):
        ss = []
        for key in self.kwds:
            ss.append(key + ' = ' + repr(getattr(self, key)))

        return '\n'.join(ss)


DEBUG = 0



def pvalue_normal(w, r, n1, n2, kind='min'):
    """Approximate probality and related properties for precedence test

    """
    if kind == 'min':
        a = r
    elif kind == 'max':
        a = n2
    elif kind == 'mean':
        a = 0.5 * (r + n2)
    else:
        raise ValueError('kind not recognized')


    v10 = n1 * r / (n1 + 1.)
    #v01 = 0.5 * (r - 1) * v10   # ?? typo v01
    v01 = 0.5 * r * (r + 1) * n1 / (n2 + 1)
    v20 = v10  / (n2 + 2.) * (2 * n1 + n2 + (r - 1) * (n1 - 1))
    v11 = 0.5 * (r + 1) * v20
    v02 = v10 * (r + 1) / 12. / (n2 + 2.) * (
                2 * (2*r + 1)*(2*n1 + n2) + (n1 - 1) * (r - 1) * (3*r + 2))
    mean = 0.5 * n1 * (n1 + 2 * a + 1) - (a + 1) * v10 + v01
    #m1 =  0.5 * n1 * (n1 + 2 * a + 1) - (a + 1) * v10
    #v02, v20, v11 = np.multiply([v02, v20, v11], 2)
    mom2nc = (0.5 * n1 * (n1 + 2 * a + 1))**2 + (a + 1)**2 * v20 + v02 - \
             2 * (a + 1) * v11 + n1 * (n1 + 2 * a + 1) * (v01 - (a + 1) * v10)
    var = mom2nc - mean**2
    zstat = (w - mean + 0.5) / np.sqrt(var)

    res = Holder(zstat=zstat,
                 pvalue=stats.norm.cdf(zstat),
                 mean=mean,
                 var=var,
                 mom2nc=mom2nc,
                 v=(v10, v01, v20, v02, v11),
                 a = a
                 )
    return res


def mmoments(nobs1, nobs2, r_cut):
    n1, n2, r = nobs1, nobs2, r_cut
    v10 = n1 * r / (n1 + 1.)
    v01 = 0.5 * r * (r + 1) * n1 / (n2 + 1)   # ?? typo v01
    v20 = v10  / (n2 + 2.) * (2 * n1 + n2 + (r - 1) * (n1 - 1))
    v11 = 0.5 * (r + 1) * v20
    v02 = v10 * (r + 1) / 12. / (n2 + 2.) * (
                2 * (2*r + 1)*(2*n1 + n2) + (n1 - 1) * (r - 1) * (3*r + 2))

    return v10, v01, v20, v02, v11

def wmoments_from_mmoments(m_moms, n1, r_cut, n2, kind='min'):
    v10, v01, v20, v02, v11 = m_moms
    r = r_cut
    if kind == 'min':
        a = r
    elif kind == 'max':
        a = n2
    elif kind == 'mean':
        a = 0.5 * (r + n2)
    else:
        raise ValueError('kind not recognized')
    mean = 0.5 * n1 * (n1 + 2 * a + 1) - (a + 1) * v10 + v01
    #m1 =  0.5 * n1 * (n1 + 2 * a + 1) - (a + 1) * v10
    #v02, v20, v11 = np.multiply([v02, v20, v11], 2)
    mom2nc = (0.5 * n1 * (n1 + 2 * a + 1))**2 + (a + 1)**2 * v20 + v02 - \
             2 * (a + 1) * v11 + n1 * (n1 + 2 * a + 1) * (v01 - (a + 1) * v10)
    var = mom2nc - mean**2
    return mean, var, mom2nc


########  exact solution

# written by Oleksandr Huziy
def bin_allocations(nobs, nbins):
    #my recursive solution with memoization

    def enumit_my(nobs, rr, memo):

        key = (nobs, rr)
        if key in memo:
            return memo[key]

        if nobs == 0:
            return [rr * [0, ], ]
        if rr == 1:
            return [[i, ] for i in range(nobs + 1)]
        res = []
        for i1 in range(nobs + 1):
            res += [[i1,] + vec for vec in enumit_my(nobs - i1, rr - 1, memo)]

        memo[key] = res
        return res

    memo = {}
    result = enumit_my(nobs, nbins, memo)
    return result

def prob_between(s, r, n1, n2):
    num = comb(n1 + n2 - s - r, n2 - r)
    denom = comb(n1 + n2, n2)
    return num * 1. / denom

class PrecedenceBase(object):
    """Base class for common methods of Wilcoxon-type Precedence distribution

    """

    @cache_readonly
    def bin_sum(self):
        return self.bin_allocation().sum(1)

    @cache_readonly
    def bin_isum(self):
        m_all = self.bin_allocation()
        return np.dot(m_all, np.arange(1, self.nbins + 1))

    def precedence_statistic(self, msum=None, misum=None, kind='min'):
        if msum is None:
            msum = self.bin_sum
        if misum is None:
            misum = self.bin_isum
        n1, r, n2 = self.nobs, self.nbins, self.nobs2
        res = []
        if kind in ['all', 'min']:
            wmin = 0.5 * n1 * (n1 + 2 * r + 1) - (r + 1) * msum + misum
            res.append(wmin)
        if kind in ['all', 'max']:
            wmax = 0.5 * n1 * (n1 + 2 * n2 + 1) - (n2 + 1) * msum + misum
            res.append(wmax)
        if kind in ['all', 'mean']:
            a = 0.5 * (r + n2)
            wmean = 0.5 * n1 * (n1 + 2 * a + 1) - (a + 1) * msum + misum
            res.append(wmean)

        if len(res) == 1:
            return res[0]
        else:
            return tuple(res)



class PrecedenceDistribution(object):
    """Exact Wilcoxon-type Precedence distribution

    ..Warning:: This evaluates a very large number of cases for large
        nobs. If the number of cases is to large then a MemoryError will be
        raised in the current implementation.

    """

    def __init__(self, nobs, nbins, nobs2):
        self.nobs = nobs
        self.nobs2 = nobs2
        self.nbins = nbins
        self.n_support = comb(nobs + nbins, nbins)
        if self.n_support > 1e6:
            import warnings
            warnings.warn('more than 1 million cases.', UserWarning)

        # initialize other attributes (this quiets pylint)
        self.m_all = None
        self._pmf = None

    @property
    def bin_allocation(self):
        # for now caching without decorator
        if self.m_all is None:
            self.m_all = np.asarray(bin_allocations(self.nobs, self.nbins))
        return self.m_all

    @cache_readonly
    def bin_sum(self):
        return self.bin_allocation.sum(1)

    @cache_readonly
    def bin_isum(self):
        m_all = self.bin_allocation
        return np.dot(m_all, np.arange(1, self.nbins + 1))

    # TODO: cached attribute
    def pmf_bin_allocation(self):
        if self._pmf is None:
            m_all = self.bin_sum
            self._pmf = prob_between(m_all, self.nbins, self.nobs, self.nobs2)
        return self._pmf

    @cache_readonly
    def moments_sum(self):
        msum = self.bin_sum
        misum = self.bin_isum
        probs = self.pmf_bin_allocation()
        v10 = np.dot(msum, probs)
        v01 = np.dot(misum, probs)
        v20 = np.dot(msum**2, probs)
        v02 = np.dot(misum**2, probs)
        v11 = np.dot(msum * misum, probs)
        return (v10, v01, v20, v02, v11)

    def precedence_statistic(self, msum=None, misum=None, kind='min'):
        if msum is None:
            msum = self.bin_sum
        if misum is None:
            misum = self.bin_isum
        n1, r, n2 = self.nobs, self.nbins, self.nobs2
        res = []
        if kind in ['all', 'min']:
            wmin = 0.5 * n1 * (n1 + 2 * r + 1) - (r + 1) * msum + misum
            res.append(wmin)
        if kind in ['all', 'max']:
            wmax = 0.5 * n1 * (n1 + 2 * n2 + 1) - (n2 + 1) * msum + misum
            res.append(wmax)
        if kind in ['all', 'mean']:
            a = 0.5 * (r + n2)
            wmean = 0.5 * n1 * (n1 + 2 * a + 1) - (a + 1) * msum + misum
            res.append(wmean)

        if len(res) == 1:
            return res[0]
        else:
            return tuple(res)

    def precedence_properties(self, msum=None, misum=None, kind='min', sk=False):
        # TODO: add skew and kurtosis
        w = self.precedence_statistic(msum=None, misum=None, kind=kind)
        probs = self.pmf_bin_allocation()
        m1 = np.dot(w, probs)
        m2 = np.dot(w**2, probs)
        var = m2 - m1**2

        if not sk:
            return m1, var, m2
        else:
            # central 3rd and 4th moment
            mc3 = np.dot((w - m1)**3, probs)
            mc4 = np.dot((w - m1)**4, probs)
            skew = mc3 / var**(3./2)
            kurt = mc4 / var**2     # no --3   not relative to normality
            return m1, var, skew, kurt, m2, mc3, mc4

    def pvalue(self, w_stat, kind='min'):
        # TODO we don't cache yet the precedence_statistic
        w = self.precedence_statistic(msum=None, misum=None, kind=kind)
        probs = self.pmf_bin_allocation()
        mask = (w <= w_stat)
        pvalue = probs[mask].sum()
        return pvalue

    def precedence_test(self, bin_allocation, kind='min'):
        m = bin_allocation
        msum = m.sum(-1)
        misum = (m * np.arange(1, self.nbins + 1)).sum(-1)
        w_stat = self.precedence_statistic(msum=msum, misum=misum, kind=kind)
        p_value = self.pvalue(w_stat, kind=kind)
        return w_stat, p_value


########## end exact
########## Permutation based

class PrecedencePermutationDistribution(PrecedenceBase):
    """


    note: bin_allocation is an attribute here

    """

    def __init__(self, nobs, nbins, nobs2):
        self.nobs = nobs
        self.nobs2 = nobs2
        self.nbins = nbins
        self.n_support = comb(nobs + nbins, nbins)
        self.bin_allocation = None


    def run_permutation(self, n_rep=10000):
        # TODO: we want to optimize a bit on storage
        # which information do we keep, what dtype do we use
        rr = self.nbins
        n1, n2 = self.nobs, self.nobs2
        # I'm permuting the ranks,
        # (a better alternative might be permuting the group indices)
        x = np.arange(n1 + n2, dtype='int32') #
        m_all = []
        for _ in xrange(n_rep):
            np.random.shuffle(x)
            # Note we are truncating to rr
            m_all.append(get_count(x[n1:], x[:n1])[:rr])

        # store or append results
        if self.bin_allocation is None:
            self.bin_allocation = np.asarray(m_all)
        else:
            #n_rep_old = len(self.bin_allocation)
            self.bin_allocation = np.append(self.bin_allocation, m_all, 0)
            # TODO: reset and check stored attributes for updating
            # we should get a pattern for resampling with updating
            # for now delete to force recalculation, a waste and fragile
            del self._cache['bin_sum']
            del self._cache['bin_isum']
            # basic check for short arrays without prespecifying
            n_rep_stored = len(self.bin_allocation)
            for key, item in self._cache.items():
                try:
                    n = len(item)
                    if n == (n_rep_stored - n_rep):
                        import warnings
                        warnings.warn('cache for %s maybe out of sync' % key)
                except TypeError:
                    pass


    @cache_readonly
    def bin_sum(self):
        return self.bin_allocation.sum(1)

    @cache_readonly
    def bin_isum(self):
        m_all = self.bin_allocation
        return np.dot(m_all, np.arange(1, self.nbins + 1))

    @cache_readonly
    def moments_sum(self):
        # wrong version use this for bootstrap or permutation version
        msum = self.bin_sum
        misum = self.bin_isum
        v10 = msum.mean(0)
        v01 = misum.mean()
        v20 = (msum**2).mean()
        v02 = (misum**2).mean()
        v11 = (msum*misum).mean()
        return (v10, v01, v20, v02, v11)

    def precedence_properties(self, msum=None, misum=None, kind='min', sk=False):
        # TODO: add skew and kurtosis
        w = self.precedence_statistic(msum=None, misum=None, kind=kind)
        m1 = w.mean()
        m2 = (w**2).mean()
        var = m2 - m1**2


        if not sk:
            return m1, var, m2
        else:
            # TODO: use scipy.stats.describe instead,
            #  better estimator of skew, kurt
            # central 3rd and 4th moment
            mc3 = ((w - m1)**3).mean()
            mc4 = ((w - m1)**4).mean()
            skew = mc3 / var**(3./2)
            kurt = mc4 / var**2     # no --3   not relative to normality
            return m1, var, skew, kurt, m2, mc3, mc4


    def pvalue(self, w_stat, kind='min'):
        # TODO we don't cache yet the precedence_statistic
        w = self.precedence_statistic(msum=None, misum=None, kind=kind)
        mask = (w <= w_stat)
        pvalue = mask.mean(0)   # TODO check broadcasting for vectorized
        return pvalue

########## Data analysis

def get_count(x1, x0):
    """count the number of occurencies in first array between observations of
    second array


    """

    x1 = np.sort(x1)
    x0 = np.sort(x0)
    rank = np.searchsorted(x1, x0)
    m = np.diff(np.concatenate(([0], rank)))   # M
    return m


class PrecedenceResults(Holder):

    def __str__(self):
        return self.summary()

    def summary(self):
        res_mat = np.column_stack((self.w_min, self.pvalue_min,
                                   self.w_max, self.pvalue_max,
                                   self.w_mean, self.pvalue_mean))
        if np.squeeze(res_mat).ndim > 1:
            # Todo need table
            return res_mat
        else:
            res_mat = np.squeeze(res_mat)

        ss = \
        '''
        wmin  = %6.2F, pval = %6.4F
        wmax  = %6.2f, pval = %6.4F
        wmean = %6.2f, pval = %6.4F
        ''' % tuple(res_mat)
        #(self.w_min, self.pval_min, self.w_max, self.pval_max,
         #      self.w_mean, self.pval_mean)
        return ss

def precedence(x1, x2, r_cut, nobs1, nobs2, kind='min', pvalue_method='auto'):
    """Wilcoxon-type Precedence Tests

    We test whether two samples come from the same distribution against a
    one-sided alternative that the cumulative distribution function of the
    first sample is larger than the one of the second sample. ::

        H0: F1 == F2
        H1: F1 > F2

    Where F1 is the cumulative distribution of the type of observations in the
    first sample, and F2 the one of the second sample.

    This test does not take ties into account.

    We observe n1 units of the first type and n2 units of the second type.
    The observations are only available up to the r_cut'th observation of the
    second type. The test is based on the number of observations in the first
    sample before we observe the r_cut'th event (observation) of the second
    sample.

    Parameters
    ----------
    x1, x2 : array_like
        two samples to be tested. Both need to be 1-D.
    r_cut : int
        number of observations observed in the second sample.
    nobs1, nobs2 : int
        number of units in the experiment, nobs1 of first type, nobs2 of second
        type.
    kind : string
        not used, currently we always calculate all three statistics: 'min',
        'max' and 'mean'.
    pvalue_method : string in ['auto', 'normal', 'exact', 'permutation']
        which method to use for the p-values, for calculating or approximating
        the distribution of the test statistic.
        The p-value based on the normal approximation will always be calculated,
        exact and permutation p_values if requested.
        'auto' is currently 'normal'

        ..Warning:: `exact` evaluates a very large number of cases for large
        nobs1. If the number of cases is to large then a MemoryError will be
        raised in the current implementation.

    Returns
    -------
    results : instance of PrecedenceResults, with attached attributes.
        The main attributes are the test statistic `w_min`, `w_max` and `w_mean`
        and the associated p-values `pvalue_min`, `pvalue_max` and
        `pvalue_mean`. `results` also has a summary method to print the
        main information.


    """
    n1 = nobs1 #len(x1)
    n2 = nobs2 #len(x2)
    m_counts = get_count(x1, x2)
    m_counts = np.atleast_2d(m_counts)
    msum = m_counts.cumsum(1)
    misum = (m_counts * np.arange(1, m_counts.shape[1] + 1)).cumsum(1)
    #s = msum  # Notation in article NZ
    #s_star = misum

    # Calculate the Wilcoxon-type precedence statistic
    w = lambda a: 0.5 * n1 * (n1 + 2 * a + 1) - (a + 1) * msum + misum
    w_min = w(r_cut)
    w_max = w(n2)
    w_mean = w(0.5 * (r_cut + n2))
    #temporary
    w_min = w_min[:, r_cut-1]
    w_max = w_max[:, r_cut-1]
    w_mean = w_mean[:, r_cut-1]
    #r_cut_all = np.arange(1, m_counts.shape[1] + 1)

    # we always calculate the normal approximation p-value, it's cheap
    pval_min = pvalue_normal(w_min, r_cut, n1, n2, kind='min').pvalue
    pval_max = pvalue_normal(w_max, r_cut, n1, n2, kind='max').pvalue
    pval_mean = pvalue_normal(w_mean, r_cut, n1, n2, kind='mean').pvalue

    res_kwd = {}  # dictionary for attributes of results class
    res_kwd['pvalue_min_normal'] = pval_min
    res_kwd['pvalue_max_normal'] = pval_max
    res_kwd['pvalue_mean_normal'] = pval_mean

    # set the default p-value
    if pvalue_method in ['normal', 'auto']:
        # TODO: use exact if 'auto' and n1 is small
        res_kwd['pvalue_min'] = res_kwd['pvalue_min_normal']
        res_kwd['pvalue_max'] = res_kwd['pvalue_max_normal']
        res_kwd['pvalue_mean'] = res_kwd['pvalue_mean_normal']

    elif pvalue_method == 'permutation':
        pd = PrecedencePermutationDistribution(n1, r_cut, n2)
        pval_min_exact = pd.pvalue(w_min, kind='min')
        pval_max_exact = pd.pvalue(w_max, kind='max')
        pval_mean_exact = pd.pvalue(w_mean, kind='mean')
        # attach it under separate name
        res_kwd['pvalue_min'] = res_kwd['pvalue_min_permutation'] = pval_min
        res_kwd['pvalue_max'] = res_kwd['pvalue_max_permutation'] = pval_max
        res_kwd['pvalue_mean'] = res_kwd['pvalue_mean_permutation'] = pval_mean

    elif pvalue_method == 'exact':
        # do exact last if we want to add pvalue_method='all'
        # with exact as default
        pd = PrecedenceDistribution(n1, r_cut, n2)
        pval_min_exact = pd.pvalue(w_min, kind='min')
        pval_max_exact = pd.pvalue(w_max, kind='max')
        pval_mean_exact = pd.pvalue(w_mean, kind='mean')
        # attach it under separate name
        res_kwd['pvalue_min'] = res_kwd['pvalue_min_exact'] = pval_min_exact
        res_kwd['pvalue_max'] = res_kwd['pvalue_max_exact'] = pval_max_exact
        res_kwd['pvalue_mean'] = res_kwd['pvalue_mean_exact'] = pval_mean_exact

    res = PrecedenceResults( w_min=w_min,
                             w_max=w_max,
                             w_mean=w_mean,
                             msum=msum,
                             misum=misum,
                             m_counts=m_counts,
                             **res_kwd)

    return res

