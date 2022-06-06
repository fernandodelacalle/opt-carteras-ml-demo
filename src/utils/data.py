import numpy as np
import pandas as pd



def subset_by_dates(vobject, start_date, end_date=None, prune_empty=True):
    data_subset = None
    if isinstance(vobject, pd.DataFrame) or isinstance(vobject, pd.Series):
        if end_date is None:
            data_subset = vobject.loc[start_date:].copy()
        else:
            data_subset = vobject.loc[start_date:end_date].copy()

    if isinstance(vobject, dict):
        if end_date is None:
            data_subset = {
                key: df.loc[start_date:].copy()
                for key, df in vobject.items()
            }
        else:
            data_subset = {
                key: df.loc[start_date:end_date].copy()
                for key, df in vobject.items()
            }
        if prune_empty:
            data_subset = {
                key: df for key, df in data_subset.items() if df.shape[0] > 0
            }
    return data_subset



def _fill_stockdata_field(stock_serie, merged_serie):
    """
    auxiliary function for market_homo_data function
    """
    start_date = stock_serie.index[0]
    end_date = stock_serie.index[-1]
    fill_serie = merged_serie.loc[start_date:end_date].fillna(method='ffill')
    return fill_serie.reindex(merged_serie.index)

def market_homo_data(stock_data, field='close'):
    """
    builds a dataframe with a particular field of all OHLCV stock data.
    These series are synchronized and then the gaps are filled 
    with previous data (eg. price), but only within eacg series.  
    NaNs at the beginning and at the end are preserved
    
    Args:
        stock_data: dictionary of OHLCV dataframes
        field: the name of field. Default 'close'. Choose between 
        ['open','high','low','close','vol']
    Returns
        synchronized df with the fields from all stock data
    """
    
    stock_field = pd.DataFrame({tk: df[field] 
                                for tk, df in stock_data.items()})
    
    cleanfill_dict = {tk: _fill_stockdata_field(df, stock_field[tk])
                      for tk, df in stock_data.items()}
    return pd.DataFrame(cleanfill_dict)
