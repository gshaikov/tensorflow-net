# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 22:25:57 2017

@author: gshai
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


class DataContainer(object):
    '''DataContainer'''

    def __init__(self, data):
        self.data = data

        self.train_size = None
        self.train_images = None
        self.train_labels = None

        self.dev_size = None
        self.dev_images = None
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

    def split_train_dev(self, train_size, dev_size):
        '''split_train_dev'''
        self.train_size = train_size
        self.dev_size = dev_size

        # create images and labels arrays
        # convert each 0...9 label to [0,1,0,...,0] vector
        images = self.data[:, 1:]
        images = self.array_to_img(images)

        labels = self.data[:, :1]
        labels = OneHotEncoder(sparse=False).fit_transform(labels)

        # split arrays into train and dev
        self.train_images, self.dev_images = self._split_array(
            images, self.train_size, self.dev_size)
        self.train_labels, self.dev_labels = self._split_array(
            labels, self.train_size, self.dev_size)

    @staticmethod
    def array_to_img(array):
        '''
        array_to_img
        shape of array -- [# of examples, # features]
        '''
        n_examples = array.shape[0]
        images = np.reshape(array, [n_examples, 28, 28])
        images = images[:, :, :, np.newaxis]
        return images

    @staticmethod
    def _split_array(data_array, *args):
        '''dataset of variable size data arrays'''
        if args:
            sets = list()
            start = 0
            for size in args:
                sets.append(data_array[start:start + size])
                start = size
        else:
            sets = [data_array]
        assert isinstance(sets, list)
        return sets

    def shuffle_train(self):
        '''shuffle_train'''
        reorder = list(np.random.permutation(self.train_size))
        self.train_images = self.train_images[reorder, :, :, :]
        self.train_labels = self.train_labels[reorder, :]

    def get_train_batches(self, minibatches_size):
        '''
        get_train_batches
        '''
        n_minibatches = np.floor(
            self.train_size / minibatches_size).astype('int')
        last_minibatch = self.train_size - n_minibatches * minibatches_size

        # create batches
        batches_train_images = self._create_batches(
            self.train_images, n_minibatches, last_minibatch)

        batches_train_labels = self._create_batches(
            self.train_labels, n_minibatches, last_minibatch)

        # put feature and label batches together
        batches_train = list(zip(batches_train_images, batches_train_labels))
        return batches_train

    @staticmethod
    def _create_batches(dataset, n_minibatches, last_minibatch):
        '''_create_batches'''
        if last_minibatch:
            dataset_batches = np.split(
                dataset[:-last_minibatch], n_minibatches, axis=0)
            dataset_batches.append(dataset[-last_minibatch:])
        else:
            dataset_batches = np.split(dataset, n_minibatches, axis=0)
        return dataset_batches


# if __name__ == '__main__':
#     dataset_train = DataContainer.get_dataset(
#         'dataset/train.csv', shuffle=True)

#     train_size = 12
#     dev_size = 5
#     dataset_train.split_train_dev(train_size, dev_size)

#     dataset_train.shuffle_train()

#     minibatches_size = 4
#     output = dataset_train.get_train_batches(minibatches_size)

#     dataset_test = DataContainer.get_dataset('dataset/test.csv')
