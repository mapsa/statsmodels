*insheet using "E:\Josef\eclipsegworkspace\statsmodels-git\statsmodels-all-new\statsmodels\statsmodels\datasets\kmenta\kmenta.csv", double comma
reg3 (consump price income) (consump price farmprice trend)

tempname filename
local filename = "kmenta_3sls_stata.py"

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
*clear cov
*clear params_table
*local matrix cov = e(V)
matrix cov = e(V)
*svmat cov, names(cov)
matrix params_table = r(table)'
*svmat params_table, names(params_table)

estmat2nparray params_table cov, saving(`filename') format("%16.0g") replace
/*------------------*/

reg3 (consump price income) (consump price farmprice trend), endog(price) 2sls

tempname filename
local filename = "kmenta_2sls_stata.py"

/* boiler plate, add matrices if needed */
tempname cov
tempname params_table
*clear cov
*clear params_table
local matrix cov = e(V)
*svmat cov, names(cov)
matrix params_table = r(table)'
*svmat params_table, names(params_table)

estmat2nparray params_table cov, saving(`filename') format("%16.0g") replace
/*------------------*/
