import pandas as pd
import numpy as np

#from port_opt.cla import CriticalLineAlgorithm
#from port_opt.cla_orig import CriticalLineAlgorithm
from port_opt.cla_limited import CriticalLineAlgorithm
from port_opt.hpr import HprAlgorithm


class PortfolioOptimizer:
    
    def __init__(self, returns):
        self.returns = returns
        self.mu = returns.mean()
        self.cov = returns.cov()
        self.corr = returns.corr()

    def cla_max_sharpe_portfolio(self, min_w=0, max_w=1):
        weight_bounds=(min_w, max_w)
        cla = CriticalLineAlgorithm(weight_bounds=weight_bounds)
        cla.allocate(expected_asset_returns=np.squeeze(self.mu.values),
                     covariance_matrix=self.cov.values,
                     asset_names=self.cov.columns,
                     solution='max_sharpe')
        # Maximum Sharpe Solution
        max_sharpe_w = cla.weights.sort_values(by=0, ascending=False, axis=1)
        max_sharpe_w = max_sharpe_w[max_sharpe_w > 0].dropna(axis=1)
        max_sharpe_w = max_sharpe_w.iloc[0]
        max_sharpe_w.name = 'max_sharpe_cla'
        return max_sharpe_w

    def cla_min_volatility_portfolio(self, min_w=0, max_w=1):
        weight_bounds=(min_w, max_w)
        cla = CriticalLineAlgorithm(weight_bounds=weight_bounds)
        # Minimum Variance Solution
        cla.allocate(expected_asset_returns=np.squeeze(self.mu.values),
                     covariance_matrix=self.cov.values,
                     asset_names=self.cov.columns,
                     solution='min_volatility')
        min_variance_w = cla.weights.sort_values(by=0, ascending=False, axis=1)
        min_variance_w = min_variance_w[min_variance_w > 0].dropna(axis=1)
        min_variance_w = min_variance_w.iloc[0]
        min_variance_w.name = 'min_volatility_cla'
        return min_variance_w

    def hrp_portfolio(self):
        hpr_algo = HprAlgorithm(self.cov, self.corr)
        hpr = hpr_algo.allocate()
        hpr.name = 'hpr'
        return hpr

    def ivp_portfolio(self):
        ivp = 1. / np.diag(self.cov)
        ivp /= ivp.sum()
        ivp = pd.Series(ivp, index=self.cov.index)
        ivp = ivp.sort_values(ascending=False)
        ivp.name = 'ivp'
        return ivp

    def eq_w_portfolio(self):
        eqw = pd.Series(index=self.returns.columns)
        n = len(self.returns.columns)
        eqw.loc[:] = 1/n
        eqw.name = 'eqw'
        return eqw

    def cla_efficient_frontier(self, min_w=0, max_w=1):
        weight_bounds=(min_w, max_w)
        cla = CriticalLineAlgorithm(weight_bounds=weight_bounds)
        # Efficient Frontier Solution
        cla.allocate(expected_asset_returns=np.squeeze(self.mu.values),
                    covariance_matrix=self.cov.values,
                    asset_names=self.cov.columns,
                    solution='efficient_frontier')
        means_cla, sigma_cla = cla.efficient_frontier_means, cla.efficient_frontier_sigma
        return means_cla, sigma_cla

    def cla_max_sharpe_points(self, min_w=0, max_w=1):
        weights = self.cla_max_sharpe_portfolio(min_w=min_w, max_w=max_w)
        weights = weights.reindex(self.returns.columns).fillna(0)
        ret, risk = self._compute_ret_sigma(weights)
        return ret, risk, weights

    def _compute_ret_sigma(self, w):
        r = w.dot(self.mu)
        std = np.sqrt(w.dot(self.cov).dot(w))
        return r, std
