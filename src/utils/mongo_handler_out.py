import os

import pymongo
import pandas as pd


class MongoHandler:
    def __init__(self,
                 database_name):
        url = os.environ['MONGO']
        self.client = pymongo.MongoClient(url)
        self.db = self.client[database_name]

    def get_stock_data(self, collection_name):
        collection = self.db[collection_name]
        documents = collection.find()
        data = dict()
        for doc in documents:
            ticker = doc['ticker']
            df = pd.read_json(doc['data'])
            if not df.empty:
                data[ticker] = df
        return data
