import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np


class Data:
    def __init__(self, data_path, max_genres=3, batch_size=32):
        self.data_path = './cleaned-data/' + data_path
        self.max_genres = max_genres
        self.batch_size = batch_size

    def read_csv(self):
        df = pd.read_csv(self.data_path)

    def config_column_genres_in_dict(self, df_dict):
        gen = df_dict.pop('genres', None)
        new_gen = []
        for i in gen:
            tmp = i.split('|')
            while len(tmp) > self.max_genres:
                tmp.pop()
            while len(tmp) < self.max_genres:
                tmp.append('null')
            new_gen.append(tmp)
        df_dict['genres'] = new_gen
        return df_dict

    # A utility method to create a tf.data dataset from a Pandas data_frame
    def df_to_dataset(self, data_frame, shuffle=True):
        data_frame = data_frame.copy()
        labels = data_frame.pop('rating')
        df_dict = self.config_column_genres_in_dict(data_frame.to_dict(orient='list'))
        ds = tf.data.Dataset.from_tensor_slices((df_dict, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(data_frame))
        ds = ds.batch(self.batch_size)
        return ds

    def get_data(self):
        data_frame = pd.read_csv(self.data_path).head(100)
        train, test = train_test_split(data_frame, test_size=0.2)
        train, val = train_test_split(data_frame, test_size=0.25)
        print(len(train))
        print(len(val))
        print(len(test))
        train_ds = self.df_to_dataset(train)
        val_ds = self.df_to_dataset(val, shuffle=False)
        test_ds = self.df_to_dataset(test, shuffle=False)

        return train_ds, val_ds, test_ds


if __name__ == '__main__':
    data = Data('ml-1m.csv')
    train_ds, val_ds, test_ds = data.get_data()
    print(type(train_ds))
    for feature_batch, label_batch in test_ds.take(1):
        # print('Every feature:', list(feature_batch.keys()))
        # print('A batch of genres:', feature_batch['genres'])
        # test = np.concatenate([test, label_batch.numpy()], axis=0)

        print(feature_batch)
