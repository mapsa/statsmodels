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
        y = x[:, :k_vars-1].sum(1) + np.random.randn(nobs)

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
        for k in range(stols.k_vars_x-1): #keep one for next test
            #store anticipated results
            params_new = stols.params_new().copy()
            rss_new = stols.rss + stols.rss_diff()
            stols.sweep(k)
            res = self.cached_ols(stols.is_exog)
            assert_equal(params_new[k], stols.params)
            assert_almost_equal(params_new[k], res.params, decimal=13)
            assert_almost_equal(rss_new, stols.rss, decimal=13)
            assert_almost_equal(stols.rss, res.rss, decimal=13)


if __name__ == '__main__':
    TestSweep.setup_class()
    tt = TestSweep()
    tt.test_sequence()
