import numpy as np
import pandas as pd
import cvxpy as cp

#from port_opt.cla import CriticalLineAlgorithm
#from port_opt.cla_orig import CriticalLineAlgorithm
from port_opt.cla_limited import CriticalLineAlgorithm



def efficient_frontier_convex(returns, n_samples=1000, gamma_low=-1, gamma_high=10):
    if isinstance(returns, (np.ndarray, np.generic)):
        sigma = np.cov(returns.T)
        mu = np.mean(returns, axis=0)
        print(sigma.shape)
        print(mu.shape)

    if isinstance(returns, (pd.Series, pd.DataFrame)):
        sigma = returns.cov().values
        mu = np.mean(returns, axis=0).values  

    n = sigma.shape[0]        
    w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)
    ret = mu.T*w
    risk = cp.quad_form(w, sigma)
    prob = cp.Problem(cp.Maximize(ret - gamma*risk), 
                      [cp.sum(w) == 1,  w >= 0]) 
    risk_data = np.zeros(n_samples)
    ret_data = np.zeros(n_samples)
    gamma_vals = np.logspace(gamma_low, gamma_high, num=n_samples)

    portfolio_weights = []    
    for i in range(n_samples):
        gamma.value = gamma_vals[i]
        prob.solve()
        risk_data[i] = np.sqrt(risk.value)
        ret_data[i] = ret.value
        portfolio_weights.append(w.value)   
    return ret_data, risk_data, gamma_vals, portfolio_weights


def efficient_frontier_cla(returns, min_w=0, max_w=1):
    
    
    weight_bounds=(min_w, max_w)
    
    mu = returns.mean()
    cov = returns.cov()
    cla = CriticalLineAlgorithm(weight_bounds=weight_bounds)
    # Efficient Frontier Solution
    cla.allocate(expected_asset_returns=np.squeeze(mu),
                 covariance_matrix=cov,
                 asset_names=cov.columns,
                 solution='efficient_frontier')
    means_cla = cla.efficient_frontier_means
    sigma_cla = cla.efficient_frontier_sigma
    weights_cla = cla.weights
    
    return means_cla, sigma_cla, weights_cla



