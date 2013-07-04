
===============================
What's new in statsmodels 0.5.0
===============================

Enhancements to existing Models
===============================

Formulas
--------

Statsmodels now has R-style formulas based on patsy which is available on pypi. Instead of building an array of explanatory variables, the design matrices, using numpy, users can now write a formula that automatically constructs the design matrix from the data.

A simple example with two explanatory variables and a constant using a pandas DataFrame to hold the data is as simple as ::


    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf

    # Load data
    url = 'http://vincentarelbundock.github.io/Rdatasets/csv/HistData/Guerry.csv'
    dat = pd.read_csv(url)

    # Fit regression model (using the natural log of one of the regressors)
    results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()

The use of formulas is especially convenient when some explanatory variables are categorical variables because patsy will automatically encode them in a appropriate dummy or contrast matrix. Formulas can use interaction terms like

`y ~ gender + race + gender : race` or

`y ~ gender * race`

Formulas can also be used after the estimation in `t_test` for testing linear restrictions on the parameters and in `predict` to calculate the prediction for a new set of explanatory variables.


L1-penalized Discrete Models
----------------------------

A new optimization method has been added to the discrete models, which includes Logit, Probit, MNLogit and Poisson, that makes it possible to estimate the models with a L1, i.e. linear, penalization. This shrinks parameters towards zero and can set parameters that are not very different from zero to zero. This is especially useful if there are a large number of explanatory variables and a large associated number of parameters. 

Improvements to ARMA Predict
----------------------------

The prediction for the univariate time series models, AR, ARMA and ARIMA, has been improved and some limitation (or bugs) removed. 
...


Missing Value Option
--------------------

All models have now an option to check for missing values and, if necessary, to remove those observations that contain a missing value in one of the required arrays.


Presence of a Constant
----------------------

Before 0.5, results statistics were based on the assumption that a constant or intercept is present in the regression. The relevant results statistic have been adjusted so that they also produce the commonly expected results in the case when there is no intercept in the regression.

New Postestimation Results
==========================

Postestimation results are available after a model has been estimated and facilitate the analysis and interpretation of those estimation results.

Influence and Outlier Measures for OLS
--------------------------------------

After the estimation of a model with OLS, the common set of influence and outlier measures and a outlier test are now available, attached as methods to the Results instance.
dfbeta, studentized residuals, ...

Margins for Discrete Models
---------------------------

In discrete Models are the prediction or fitted values are a nonlinear function of the parameters. This makes it more difficult to interpret the marginal impact of a change in an explanatory variable, in contrast to a linear model where the impact is always given by the estimated coefficient.

Margins for discrete models have been completely rewritten and offer now additional statistics like standard errors pvalues and confidence intervals.


New Models
==========

NegativeBinomial
----------------

Negative Binomial regression has been added to the discrete models for count data. Negative Binomial has three submodels based on restriction on the extra shape parameter.
...


Quantile Regression
-------------------

Quantile Regression allows us to estimate a linear regression based on the quantiles of the residuals. It includes median regression but can also be used for quantiles that are closer to the tail of the distribution. Median regression is a robust estimator similar to the robust model RLM that has been available in statsmodels since the beginning.

Empirical Likelihood
--------------------

...

Multivariate Kernel Density and Kernel Regression
-------------------------------------------------
...

New in Statistics
=================

ANOVA based on Linear Model
---------------------------

type 1, 2, 3 sum of squares
...

Empirical Likelihood
--------------------

Statistical tests based on empirical likelihood


Power and Sample Size Calculations
----------------------------------

available for several hypothesis tests
...

New statistical hypothesis tests
--------------------------------

TOST

test for proportions

???

cohens_kappa

Tukey HSD multiple comparison enhancement, new plot


Empirical Likelihood based Hypothesis Tests
-------------------------------------------

mean, variance, ...


New and Improved Graphics
=========================

Mosaic Plot
-----------

new

Interaction Plot
----------------

new

Goodness of Fit Plots
---------------------

new and enhanced

Regression Plots
----------------

refactored and enhanced

