import scipy as sp
import numpy as np

"""
Simulates an AR(1) model using the parameters beta, c, and sigma.
Returns an nPath X nPeriod matrix, with one simulation per row.
"""
def simulate(nPath, nPeriod, beta, c, sigma):
    noise = c + sp.random.normal(0, sigma, (nPath, nPeriod))
    sims = np.zeros((nPath, nPeriod))
    sims[:,0] = noise[:,0]
    for period in range(1, nPeriod):
        sims[:,period] = beta*sims[:,period-1] + noise[:,period]
    return sims

"""
Fits an AR(1) model to the time series data ts.  AR(1) is a
linear model of the form

   x_t = beta * x_{t-1} + c + e_{t-1}

where beta is the coefficient of term x_{t-1}, c is a constant
and x_{t-1} is an i.i.d. noise term.  Here we assume that e_{t-1}
is normally distributed. 

Returns the tuple (beta, c, sigma).
"""
def fit(ts):
    # Fitting AR(1) entails finding beta, c, and the noise term.
    # Beta is well approximated by the coefficient of OLS regression
    # on the lag of the data with itself.  Since the noise term is
    # assumed to be i.i.d. and normal, we must only estimate sigma,
    # the standard deviation.

    # Estimate beta
    x = ts[0:-1]
    y = ts[1:]
    p = sp.polyfit(x,y,1)
    beta = p[0]

    # Estimate c
    c = sp.mean(ts)*(1-beta)

    # Estimate the variance from the residuals of the OLS regression.
    yhat = sp.polyval(p,x)
    variance = sp.var(y-yhat)
    sigma = sp.sqrt(variance)

    return beta, c, sigma

