import pickle

from utils.load_data_mongo import get_data
import utils.data as data

def load_all_data():
    data_all_index = {}

    stock_data, bm = get_data('ibex')
    bm = bm['ibex_div'].close
    bm.name = 'ibex_div'
    sclose = data.market_homo_data(stock_data)
    n_days = sclose.shape[0]
    data_all_index['ibex'] = [sclose, bm, n_days]

    stock_data, bm = get_data('dax')
    bm = bm['dax'].close
    bm.name = 'dax'
    sclose = data.market_homo_data(stock_data)
    n_days = sclose.shape[0]
    data_all_index['dax'] = [sclose, bm, n_days]

    stock_data, bm = get_data('stoxx')
    bm = bm['stoxx50'].close
    bm.name = 'stoxx50'
    sclose = data.market_homo_data(stock_data)
    n_days = sclose.shape[0]
    data_all_index['stoxx'] = [sclose, bm, n_days]

    stock_data, bm = get_data('stoxx_sectors')
    bm = bm['SXXR'].close
    bm.name = 'SXXR'
    sclose = data.market_homo_data(stock_data)
    n_days = sclose.shape[0]
    data_all_index['stoxx_sectors'] = [sclose, bm, n_days]
    
    return data_all_index


def save_all_data(path, data_all_index):
    with open(path, 'wb') as handle:
        pickle.dump(data_all_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_all_data_plk(path):
    with open(path, 'rb') as handle:
        data_all_index = pickle.load(handle)
    return data_all_index
