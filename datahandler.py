# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 22:25:57 2017

@author: gshai
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


class DataContainer(object):
    '''Data'''

    def __init__(self, data):
        self.data = data

        self.train_m = None
        self.train_features = None
        self.train_labels = None

        self.dev_m = None
        self.dev_features = None
        self.dev_labels = None

    @classmethod
    def get_dataset(cls, filepath, shuffle=False):
        '''
        get_dataset factory method
        https://realpython.com/blog/python/instance-class-and-static-methods-demystified/
        '''
        csv_table = pd.read_csv(filepath)
        if shuffle:
            csv_table = csv_table.sample(frac=1).reset_index(drop=True)
        csv_table = cls._normalize_df(csv_table)
        data = csv_table.as_matrix()
        return cls(data)

    @staticmethod
    def _normalize_df(pd_table):
        '''_normalize_df'''
        norm_func = (lambda x: x / 255 if not x.name == 'label' else x)
        pd_table = pd_table.apply(norm_func, axis=0)
        return pd_table

    def split_train_dev(self, train_m, dev_m):
        '''split_train_dev'''
        self.train_m = train_m
        self.dev_m = dev_m

        # create features and labels arrays
        # convert each 0...9 label to [0,1,0,...,0] vector
        features = self.data[:, 1:].T
        labels = OneHotEncoder(sparse=False).fit_transform(self.data[:, :1]).T

        # split arrays into train and dev
        self.train_features, self.dev_features = self._split_array(
            features, self.train_m, self.dev_m)
        self.train_labels, self.dev_labels = self._split_array(
            labels, self.train_m, self.dev_m)

    @staticmethod
    def _split_array(data_array, *args):
        '''dataset of variable size data arrays'''
        if args:
            sets = list()
            start = 0
            for size in args:
                sets.append(data_array[:, start:start + size])
                start = size
        else:
            sets = [data_array]
        assert isinstance(sets, list)
        return sets

    def shuffle_train(self):
        '''shuffle_train'''
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.shuffle.html
        train_array = np.hstack((self.train_features.T, self.train_labels.T))
        np.random.shuffle(train_array)
        self.train_features = train_array.T[:-10, :]
        self.train_labels = train_array.T[-10:, :]

    def get_train_batches(self, minibatches_size):
        '''
        get_train_batches
        '''
        n_minibatches = np.floor(self.train_m / minibatches_size).astype('int')
        last_minibatch = self.train_m - n_minibatches * minibatches_size

        # create batches
        batches_train_features = self._create_batches(
            self.train_features, n_minibatches, last_minibatch)

        batches_train_labels = self._create_batches(
            self.train_labels, n_minibatches, last_minibatch)

        # put feature and label batches together
        batches_train = list(zip(batches_train_features, batches_train_labels))
        return batches_train

    @staticmethod
    def _create_batches(dataset, n_minibatches, last_minibatch):
        '''_create_batches'''
        if last_minibatch:
            dataset_batches = np.split(
                dataset[:, :-last_minibatch], n_minibatches, axis=1)
            dataset_batches.append(dataset[:, -last_minibatch:])
        else:
            dataset_batches = np.split(dataset, n_minibatches, axis=1)
        return dataset_batches


if __name__ == '__main__':
    pass
#    dataset_train = Data.get_dataset('dataset/train.csv', shuffle=True)
#
#    train_m = 12
#    dev_m = 5
#    dataset_train.split_train_dev(train_m, dev_m)
#
#    dataset_train.shuffle_train()
#
#    minibatches_size = 4
#    output = dataset_train.get_train_batches(minibatches_size)
#
#    dataset_test = Data.get_dataset('dataset/test.csv')
