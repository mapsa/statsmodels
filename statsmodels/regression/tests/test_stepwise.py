# -*- coding: utf-8 -*-
"""Tests for stepwise and sequential regression

Created on Fri Jun 22 19:19:51 2012

Author: Josef Perktold
"""

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from statsmodels.regression.linear_model import OLS
from statsmodels.regression.stepwise import StepwiseOLSSweep

class TestSweep(object):

    def __init__(self):
        self.stols = StepwiseOLSSweep(self.endog, self.exog)

    @classmethod
    def setup_class(cls):
        #DGP:
        nobs, k_vars = 50, 4

        np.random.seed(85325783)
        x = np.random.randn(nobs, k_vars)
        x[:,0] = 1.   #make constant
        y = x[:, :k_vars-2].sum(1) + np.random.randn(nobs)

        cls.endog_idx = -1

        cls.endog, cls.exog = y, x
        cls.ols_cache = {}

        #cls.stols = SequentialOLSSweep(y, x)

    def cached_ols(self, is_exog):
        key = tuple(is_exog)
        if key in self.ols_cache:
            res = self.ols_cache[key]
        else:
            res = OLS(self.endog, self.exog[:, is_exog]).fit()
            self.ols_cache[key] = res
        return res



    def test_sequence(self):
        stols = self.stols
        for k in range(stols.k_vars_x-1) + [0, 1, 3, 0]: #keep one for next test
            print k
            #store anticipated results
            params_new = stols.params_new().copy()
            rss_new0 = stols.rss + stols.rss_diff(endog_idx=self.endog_idx)
            rss_new = stols.rss_new()
            assert_almost_equal(rss_new, rss_new0, decimal=13)
            #TODO: add params_full, all zeros for not included
            #this has currently rss in params full, error in test, not code
#            params_full = stols.rs_current[-1, :stols.k_vars_x]
#            assert_almost_equal(params_new, params_full + stols.params_diff(),
#                                decimal=13)

            stols.sweep(k)
            res = self.cached_ols(stols.is_exog)
            assert_equal(
                 np.squeeze(params_new[k:k+1, stols.is_exog[:stols.k_vars_x]]),
                 np.squeeze(stols.params))  #different ndim
            assert_almost_equal(params_new[k, stols.is_exog], res.params,
                                decimal=13)
            assert_almost_equal(rss_new[k], stols.rss, decimal=13)

            assert_almost_equal(stols.rss, res.ssr, decimal=12)

            if k == stols.k_vars_x - 2:  #do this once
                print repr(stols.is_exog)
                resall = tt.cached_ols(np.ones(4, bool))
                resc = tt.cached_ols(tt.stols.is_exog)
                #check fvalue of ftest
                fval3 = [resc.f_test(np.eye(3)[ii]).fvalue.item(0,0)
                            for ii in range(3)]
                fval4 = [resall.f_test(np.eye(4)[ii]).fvalue.item(0,0)
                            for ii in range(4)]
                ff = tt.stols.ftest_sweep()
                assert_almost_equal(ff[0], (fval3 + [fval4[-1]]), decimal=10)
                #check pvalue of ftest
                fpval3 = [resc.f_test(np.eye(3)[ii]).pvalue.item(0,0)
                            for ii in range(3)]
                fpval4 = [resall.f_test(np.eye(4)[ii]).pvalue.item(0,0)
                            for ii in range(4)]
                assert_almost_equal(ff[1], (fpval3 + [fpval4[-1]]), decimal=10)

        res3 = tt.stols.get_results()
        #some differences in names of attributes/properties
        attr = [('params', 'params'),
                ('bse', 'bse'),
                ('scale', 'scale2'),
                ('df_resid', 'df_resid'),
                ('nobs', 'nobs'),
                #('k_vars', 'k_vars_x'),
                ('normalized_cov_params', 'normalized_cov_params'),
                ]
        for iols, ist in attr:
            #shape mismatch params is 2d not 1d as in OLS, decide later
#             assert_almost_equal(getattr(stols, ist),
#                                 getattr(res3, iols),
#                                 decimal=12, err_msg=ist + 'differs')
            assert_almost_equal(np.squeeze(getattr(stols, ist)),
                                np.squeeze(getattr(res3, iols)),
                                decimal=12, err_msg=ist + 'differs')


class TestSweep2(TestSweep):
    #test multivariate endog
    #TDD: this will still fail in many places

    def __init__(self):
        self.stols = StepwiseOLSSweep(self.endog, self.exog)

    @classmethod
    def setup_class(cls):
        #DGP:
        nobs, k_vars = 50, 4

        np.random.seed(85325783)
        x = np.random.randn(nobs, k_vars)
        x[:,0] = 1.   #make constant
        y = x[:, :k_vars-2].sum(1) + np.random.randn(nobs)
        y = np.column_stack((y,y))

        cls.endog_idx = [-2,-1]

        cls.endog, cls.exog = y, x
        cls.ols_cache = {}



if __name__ == '__main__':
    TestSweep.setup_class()
    tt = TestSweep()
    tt.test_sequence()
    TestSweep2.setup_class()
    tt = TestSweep2()
    tt.test_sequence()
    print tt.stols.params_new()
    rr = tt.stols.rs_current
    print tt.stols.sweep(0, update=False)
    resall = tt.cached_ols(np.ones(4, bool))
    print resall.f_test(np.eye(4)[3])
    resc = tt.cached_ols(np.array([ True,  True,  True, False, False]))
    fval3 = [resc.f_test(np.eye(3)[ii]).fvalue.item(0,0) for ii in range(3)]
    fval4 = [resall.f_test(np.eye(4)[ii]).fvalue.item(0,0) for ii in range(4)]
    print fval3
    print fval4
    print tt.stols.ftest_sweep()