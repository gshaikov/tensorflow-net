# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 22:17:08 2017

@author: gshai
"""

import json

from datahandler import DataContainer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf


def _main():
    print('Neural Network with Tensorflow')

    #%%

    tf.reset_default_graph()

    # https://stackoverflow.com/questions/2835559/parsing-values-from-a-json-file
    with open('params_mode.json', 'rb') as jsonfile:
        params_mode = json.load(jsonfile)

    #%% add inputs

    features_size = 784
    labels_size = 10

    features_tf = tf.placeholder(
        tf.float32,
        shape=[features_size, None],
        name='X',
    )

    correct_labels_tf = tf.placeholder(
        tf.float32,
        shape=[labels_size, None],
        name='Y',
    )

    #%% create weights and biases

    layers_size = [features_size, 700, 400, 200, 100, 50, 25, labels_size]

    init_xavier = tf.contrib.layers.xavier_initializer()
    init_zeros = tf.zeros_initializer()

    net_params = dict()
    for idx, size in enumerate(layers_size):
        if idx != 0:
            weight = tf.get_variable(
                "W" + str(idx),
                [size, layers_size[idx - 1]],
                initializer=init_xavier
            )

            bias = tf.get_variable(
                "b" + str(idx),
                [size, 1],
                initializer=init_zeros
            )

            net_params["layer_" + str(idx)] = {
                'W': weight,
                'b': bias,
                'size': size,
            }

    #%% forward propagation

    layer_dropout = ['layer_2', 'layer_3', 'layer_4']

    # for dropout layer - probability of keping a neuron
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    A = features_tf
    for layer_name, params in net_params.items():
        Z = tf.matmul(params['W'], A) + params['b']
        A = tf.nn.relu(Z)
        if layer_name in layer_dropout:
            A = tf.nn.dropout(A, keep_prob)
    Z_L = Z

    #%% prediction evaluation nodes

    predict = tf.argmax(Z_L, axis=0)

    correct_prediction = tf.equal(
        predict,
        tf.argmax(correct_labels_tf, 0)
    )

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #%% load datasets

    # sizes of datasets
    m_train = 40000
    m_dev = 2000

    dataset = DataContainer.get_dataset('dataset/train.csv', shuffle=True)
    dataset.split_train_dev(m_train, m_dev)

    #%% parameters

    epochs = params_mode['epochs']
    minibatches_size = params_mode['minibatches_size']
    learn_rate = params_mode['learning_rate']

    lambd_val_list = params_mode['lambda_reg_list']
    keep_prob_list = params_mode['keep_prob_list']

    # for idx in range(15):
    #     rand_power = -4 * np.random.rand() - 3
    #     lambd_val_list.append(10 ** rand_power)

    # for idx in range(15):
    #     rand_value = 0.75 + np.random.rand() / 4
    #     keep_prob_list.append(rand_value)

    fig = plt.figure()
    plt.axis([
        np.min(keep_prob_list) * 0.9, np.max(keep_prob_list) * 1.1,
        np.min(lambd_val_list) * 0.9, np.max(lambd_val_list) * 1.1
    ])
    ax = plt.gca()
    ax.set_autoscale_on(False)
    plt.scatter(keep_prob_list, lambd_val_list)
    plt.yscale('log')
    plt.xlabel('keep_prob')
    plt.ylabel('lambda (log)')
    plt.show()
    plt.close(fig)

    input("Press Enter to continue...")

    #%% compute loss

    lambd = tf.placeholder(tf.float32, shape=(), name='lambda')

    logits_for_loss = tf.transpose(Z_L)
    labels_for_loss = tf.transpose(correct_labels_tf)

    loss_per_example = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits_for_loss,
        labels=labels_for_loss,
    )

    # https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    regularizer = 0
    for _, layer_params in net_params.items():
        regularizer += tf.nn.l2_loss(layer_params['W'])

    cost = tf.reduce_mean(loss_per_example + lambd * regularizer)

    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
    optimizer_step = optimizer.minimize(cost)

    #%%

    sess = tf.Session()

    saver = tf.train.Saver()

    results_final_model = pd.DataFrame(columns=[
        'model',
        'lambda',
        'keep_prob',
        'cost_train',
        'accuracy_train',
        'cost_dev',
        'accuracy_dev',
    ])

    # hyperparameter search
    exit_flag = False
    for idx_lambd, lambd_val in enumerate(lambd_val_list):
        for idx_keep_prob, keep_prob_val in enumerate(keep_prob_list):

            sess.run(tf.global_variables_initializer())

            costs_run_table = pd.DataFrame(columns=[
                'cost_train',
                'accuracy_train',
                'cost_dev',
                'accuracy_dev',
            ])

            try:
                for epoch in range(1, epochs + 1):

                    dataset.shuffle_train()
                    train_batches = dataset.get_train_batches(minibatches_size)

                    # https://stackoverflow.com/questions/25239933/how-to-add-title-to-subplots-in-matplotlib
                    # https://stackoverflow.com/questions/3823752/display-image-as-grayscale-using-matplotlib
                    # https://stackoverflow.com/questions/39659998/using-pyplot-to-create-grids-of-plots
                    fig = plt.figure()
                    ax1 = fig.add_subplot(1, 2, 1)
                    digits_train_example = train_batches[0][0][:, 0]
                    ax1.imshow(digits_train_example.reshape(
                        (28, 28)), cmap='gray')
                    ax1.set_title("train [0]")
                    ax2 = fig.add_subplot(1, 2, 2)
                    digits_train_example = dataset.dev_features[:, 0]
                    ax2.imshow(digits_train_example.reshape(
                        (28, 28)), cmap='gray')
                    ax2.set_title("dev [0]")
                    plt.show()
                    plt.close(fig)

                    cost_train_epoch = 0
                    number_of_iterations = len(train_batches)

                    for examples_features, examples_labels in train_batches:
                        # optimzie over all cost, incl. regularization
                        sess.run(optimizer_step, feed_dict={
                            features_tf: examples_features,
                            correct_labels_tf: examples_labels,
                            keep_prob: keep_prob_val,
                            lambd: lambd_val,
                        })

                        # compute only entropy part of cost, without regularization
                        cost_train_batch = sess.run(cost, feed_dict={
                            features_tf: examples_features,
                            correct_labels_tf: examples_labels,
                            keep_prob: 1.0,
                            lambd: 0,
                        })

                        cost_train_epoch += cost_train_batch / number_of_iterations

                    accuracy_train = sess.run(accuracy, feed_dict={
                        features_tf: dataset.train_features,
                        correct_labels_tf: dataset.train_labels,
                        keep_prob: 1.0,
                    })

                    # compute only entropy part of cost, without regularization
                    cost_dev_epoch, accuracy_dev = sess.run([cost, accuracy], feed_dict={
                        features_tf: dataset.dev_features,
                        correct_labels_tf: dataset.dev_labels,
                        keep_prob: 1.0,
                        lambd: 0,
                    })

                    costs_run_table = costs_run_table.append({
                        'epoch': epoch,
                        'cost_train': cost_train_epoch,
                        'accuracy_train': accuracy_train * 100,
                        'cost_dev': cost_dev_epoch,
                        'accuracy_dev': accuracy_dev * 100,
                    }, ignore_index=True)

                    print(
                        "epoch {}; cost train {:5.4f}, dev {:5.4f}; acc train {:6.4f}%, dev {:6.4f}%".format(
                            epoch,
                            cost_train_epoch,
                            cost_dev_epoch,
                            accuracy_train * 100,
                            accuracy_dev * 100))

            except KeyboardInterrupt:
                exit_flag = True

            fig = plt.figure()
            plt.plot(costs_run_table['cost_train'])
            plt.plot(costs_run_table['cost_dev'])
            plt.legend(['cost_train', 'cost_dev'])
            plt.title('Costs (epoch)')
            # https://stackoverflow.com/questions/9622163/save-plot-to-image-file-instead-of-displaying-it-using-matplotlib
            fig.savefig('results/costs_epoch.png')
            plt.show()
            plt.close(fig)

            fig = plt.figure()
            plt.plot(costs_run_table['accuracy_train'])
            plt.plot(costs_run_table['accuracy_dev'])
            plt.legend(['accuracy_train', 'accuracy_dev'])
            plt.title('Accuracy (epoch)')
            fig.savefig('results/accuracy_epoch.png')
            plt.show()
            plt.close(fig)

            # Save the variables to disk.
            # https://www.tensorflow.org/programmers_guide/saved_model
            save_path = saver.save(
                sess, "tmp/model_{:02d}_{:2d}.ckpt".format(idx_lambd, idx_keep_prob))
            print("Model saved in file: {}".format(save_path))

            results_final_model = results_final_model.append({
                'model': save_path,
                'lambda': lambd_val,
                'keep_prob': keep_prob_val,
                'cost_train': cost_train_epoch,
                'accuracy_train': accuracy_train,
                'cost_dev': cost_dev_epoch,
                'accuracy_dev': accuracy_dev,
            }, ignore_index=True)

            if exit_flag:
                break

        if exit_flag:
            break

    results_final_model.to_csv(
        'tmp/hyperparameter_search.csv',
        index=True,
    )

    #%%

    input("Press Enter to Predict...")

    #%%

    dataset_test = DataContainer.get_dataset('dataset/test.csv')
    test_features_array = dataset_test.data.T

    predictions_test = sess.run(predict, feed_dict={
        features_tf: test_features_array,
        keep_prob: 1.0,
    })

    predictions_test_df = pd.DataFrame(predictions_test).reset_index()
    predictions_test_df.columns = ['ImageId', 'Label']
    predictions_test_df['ImageId'] += 1

    for idx, row in predictions_test_df.head(10).iterrows():
        fig = plt.figure()
        plt.imshow(test_features_array[:, idx].reshape((28, 28)), cmap='gray')
        plt.show()
        print("predicted: {}\n".format(row['Label']))
        plt.close(fig)

    predictions_test_df.to_csv(
        'results/digits_test.csv',
        index=False,
    )

    #%%

    sess.close()


#%%


if __name__ == '__main__':
    _main()
