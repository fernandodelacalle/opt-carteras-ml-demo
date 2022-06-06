import pandas as pd
from pandas.tseries.offsets import BDay

from utils.mongo_handler_out import MongoHandler
from utils.data import subset_by_dates


def get_data(index: str):
    """Get data from a given index

    Parameters
    ----------
    index : str
        suported index: ibex, dax, stoxx
    cache : If true it will try to retrieve from local pickles

    Returns
    -------
    stock_data: dict
    bm: dict
    """
    if index == 'ibex':
        stock_data, _, bm = get_data_ibex()
        _ = bm.pop('ibex')
        first_day = bm['ibex_div'].index[0]

    if index == 'dax':
        stock_data, bm = get_data_dax()
        first_day = bm['dax'].index[0]

    if index == 'stoxx':
        stock_data, bm = get_data_stoxx()
        first_day = bm['stoxx50'].index[0]

    if index == 'stoxx_sectors':
        stock_data, bm = get_data_stoxx_sectors()
        first_day = bm['SXXR'].index[0]

    stock_data = subset_by_dates(stock_data, first_day)
    return stock_data, bm


def get_data_ibex():
    mongo_handler = MongoHandler('IBEX')
    data_adj = mongo_handler.get_stock_data('securities_adjusted')
    data_non_adj = mongo_handler.get_stock_data('securities_non_adjusted')
    benchmarks = mongo_handler.get_stock_data('indexes')
    return data_adj, data_non_adj, benchmarks


def get_data_stoxx_sectors():
    mongo_handler = MongoHandler('STOXX_SECTORS')
    data_adj = mongo_handler.get_stock_data('securities_adjusted')
    benchmarks = mongo_handler.get_stock_data('indexes')
    return data_adj, benchmarks


def get_data_dax():
    mongo_handler = MongoHandler('DAX')
    data_adj = mongo_handler.get_stock_data('securities_adjusted')
    benchmarks = mongo_handler.get_stock_data('indexes')
    return data_adj, benchmarks


def get_data_stoxx():
    mongo_handler = MongoHandler('EUROSTOXX')
    data_adj = mongo_handler.get_stock_data('securities_adjusted')
    benchmarks = mongo_handler.get_stock_data('indexes')
    return data_adj, benchmarks


def get_data_value_tracker(collection_id='default'):
    mongo_handler = MongoHandler('VALUE_TRACKER')
    stock_data = mongo_handler.get_stock_data(collection_id)
    return stock_data

