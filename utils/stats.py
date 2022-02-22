import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

# energy storage metric
def esm(d_truth, d_pred):
    # 
    t = esmv(d_truth)
    p = esmv(d_pred)
    return (t - p) / t

# 
def esmv(d):
    d = d / d.max()
    mv = d.mean()
    d = d - mv
    store = 0.0
    min_store = d.sum()
    for index, value in d.items():
        # Note: both subtract because value is negative in the 2nd one!
        store = store - value
        if store < min_store:
            min_store = store
    return -min_store

# could be a way of calculating predicted r2 but I don't understnad what xs is
def press_statistic(y_true, y_pred, xs):
    """
    Calculation of the `Press Statistics <https://www.otexts.org/1580>`_

    y_true  - true value (ie gas)
    y_pred  - predicted value (ie values from the regrssion line)
    xs      - independent variables ( ie heat)
    """
    res = y_pred - y_true
    hat = xs.dot(np.linalg.pinv(xs))
    den = (1 - np.diagonal(hat))
    sqr = np.square(res/den)
    return sqr.sum()

def predicted_r2(y_true, y_pred, xs):
    """
    Calculation of the `Predicted R-squared <https://rpubs.com/RatherBit/102428>`_
    """
    press = press_statistic(y_true=y_true,
                            y_pred=y_pred,
                            xs=xs
    )

    sst  = np.square( y_true - y_true.mean() ).sum()
    return 1 - press / sst
 
def r2(y_true, y_pred):
    """
    Calculation of the unadjusted r-squared, goodness of fit metric
    """
    sse  = np.square( y_pred - y_true ).sum()
    sst  = np.square( y_true - y_true.mean() ).sum()
    return 1 - sse/sst

def print_stats(heat, gas, method, nvars=1, plot=False, xl='Heat Demand (from the method)', yl='Heat Demand (from gas)'):
    # Root Mean Square Error (RMSE)
    rmse = ( ( heat - gas ) **2 ).mean() ** .5
    average = heat.mean()
    # Normalised Root Mean Square Error (nRMSE)
    nrmse = rmse / average
    # Pearsons correlation coefficient
    corr = heat.corr(gas)
    # Regression Rsquared.
    model = sm.OLS(gas.to_numpy(), heat.to_numpy())
    results = model.fit()
    p = results.params
    gradient = p[0]
    rsquared = results.rsquared
#   print('{} gradient {}'.format(method,gradient) )
#   print('params')
#   print(p)
    x = np.array([gas.min(),gas.max()])
    y = p[0] * x
    # Variance
    hvar = heat.var()
    gvar = gas.var()
    # distance between heat value and the fit line
    residual = heat - (p[0] * gas)
    # fitted value
    fit = p[0] * gas
    # check if residual and fit are the same size
    if len(residual) != len(fit):
        print('ERROR: different sized series {} {} {}'.format(method, len(residual), len(fit)))
        print(residual.index)
        print(fit.index)
        quit()
    # Fit line through the residuals - the add constant bit gives us 
    #  the intercept as well as the gradient of the fit line.
    rmodel = sm.OLS(residual.to_numpy(), sm.add_constant(fit.to_numpy()))
    residual_results = rmodel.fit()
#   print(residual_results.summary())
    res_const = residual_results.params[0]
    res_grad = residual_results.params[1]
    # adjusted R sqaured
    n = len(gas)
    k = nvars
    adjusted_rsquared = 1 - ( (1-rsquared)*(n-1) / ( n - k - 1) )
    # predicted R squared
    pr2 = predicted_r2(gas.to_numpy(), fit.to_numpy(), model.exog)
    # max
    peak = heat.max() / gas.max()
    # storage metric
    stor = esm(gas, heat)
    # output results
    print(' {0:15} {1:.2f}    {2:.2f} {3:.2f}   {4:.3f}    {5:.2f}     {6:.2f}   {7:.4f}   {8:.2f}      {9:.3f}        {10:.3f}     {11:.3f}   {12:.2f}'. format(method, corr, rmse, nrmse, rsquared, hvar, gvar, res_grad, res_const, adjusted_rsquared, pr2, peak, stor))

#   all the regression details
#   print(results.summary())

    # fit plot with the fit line
    if plot:
        plt.scatter(heat,gas,s=12)
        plt.plot(x,y,color='red')
        plt.title('Fit ' + method)
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.show()

#   this needs an r-matrix whatever that is!
#   t = results.t_test()
#   print(t)

        #   Plotting the residuals
        plt.scatter(fit,residual,s=12,color='blue')
        # Horizontal line
        y = 0 * x
        plt.plot(x,y,color='red')
        # Fit of residuals line
        x = np.array([fit.min(),fit.max()])
        y = res_const + res_grad * x
        plt.plot(x,y,color='green')
        # labels
        plt.title('Residuals vs fit ' + method)
        plt.xlabel('Fit (m * y)')
        plt.ylabel('Residual ( x - m*y )')
        plt.show()

def print_stats_header(header=' Method     '):
    print('{} Correlation RMSE NRMSE R-SQUARED Variance Gas   ResGrad   ResConst Adj-R-SQUARED Pred-R-SQUARED Max  ESM'.format(header))

def print_large_small(heat, gas, method):
    # smallest and largest diffs
    diff = gas - heat
    print(method + ' gas bigger ')
    print(diff.nsmallest())
    print(method + ' heat bigger ')
    print(diff.nlargest())

def normalize(s):
    s = s * ( 1 / s.max() )
    return s

def monthly_stats(s1,s2,name):
    errs=[]
    print('{: <8} '.format(name), end='')
    for m in range(12):
        m1 = s1[s1.index.month==m+1]
        m2 = s2[s2.index.month==m+1]
        # Root Mean Square Error (RMSE)
        rmse = ( ( m1 - m2 ) **2 ).mean() ** .5
        average = s1.mean()
        # Normalised Root Mean Square Error (nRMSE)
        nrmse = rmse / average
        print('{:.2f}  '.format(nrmse), end='' )
        errs.append(nrmse)
    print(' ')
    return errs

def monthly_stats_header():
    print('Method ', end='')
    for m in range(12):
        print('  {}   '.format(m+1), end='' )
    print(' ')
