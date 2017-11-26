# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 11:28:08 2017

@author: gshai
"""

import json

from datahandler import DataContainer

import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

import IPython.display


#%%


def conv_layer(idx, layer_a_prev, ch_out, is_training,
               filt=3, stride=1, pad='SAME'):
    '''conv_layer'''
    ch_in = layer_a_prev.get_shape().as_list()[-1]

    kernel = tf.get_variable(
        name="w" + str(idx),
        shape=[filt, filt, ch_in, ch_out],
        initializer=tf.contrib.layers.xavier_initializer(),
    )
    bias = tf.get_variable(
        name="b" + str(idx),
        shape=[1, 1, 1, ch_out],
        initializer=tf.zeros_initializer(),
    )

    layer_c = tf.nn.conv2d(
        input=layer_a_prev,
        filter=kernel,
        strides=[1, stride, stride, 1],
        padding=pad,
    )
    # layer_z = tf.layers.batch_normalization(
    #     layer_c, axis=-1, training=is_training)
    layer_z = tf.add(layer_c, bias)
    layer_a = tf.nn.relu(layer_z)
    return layer_a


def fc_layer(idx, layer_a_prev, num_units):
    '''fc_layer'''
    num_units_prev = layer_a_prev.get_shape().as_list()[-1]

    weight = tf.get_variable(
        "w" + str(idx),
        shape=[num_units_prev, num_units],
        initializer=tf.contrib.layers.xavier_initializer(),
    )
    bias = tf.get_variable(
        "b" + str(idx),
        shape=[1, num_units],
        initializer=tf.zeros_initializer(),
    )

    layer_z = tf.add(tf.matmul(layer_a_prev, weight), bias)
    layer_a = tf.nn.relu(layer_z)
    return layer_a, layer_z


def _main():
    print('Neural Network with Tensorflow')

    #%% load datasets

    # sizes of datasets
    m_train = 41000
    m_dev = 1000

    dataset = DataContainer.get_dataset('dataset/train.csv', shuffle=True)
    dataset.split_train_dev(m_train, m_dev)

    #%% parameters

    # https://stackoverflow.com/questions/2835559/parsing-values-from-a-json-file
    with open('params_mode_cnn.json', 'rb') as jsonfile:
        params_mode = json.load(jsonfile)

    epochs = params_mode['epochs']
    minibatches_size = params_mode['minibatches_size']
    learn_rate = params_mode['learning_rate']

    keep_prob_list = params_mode['keep_prob_list']

    #%% add inputs

    tf.reset_default_graph()

    img_width = 28
    img_height = 28
    labels_size = 10

    images_ph = tf.placeholder(
        tf.float32,
        shape=[None, img_height, img_width, 1],
        name='X',
    )

    correct_labels_ph = tf.placeholder(
        tf.float32,
        shape=[None, labels_size],
        name='Y',
    )

    keep_prob_ph = tf.placeholder(
        tf.float32,
        name='keep_prob'
    )

    is_training = tf.placeholder(
        tf.bool,
        name='is_training'
    )

    #%% forward propagation

    # layer 1
    layer1_a = conv_layer(
        1, images_ph, 32, is_training,
        filt=5, stride=1, pad='VALID',
    )

    layer1_p = tf.nn.max_pool(
        layer1_a,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='VALID'
    )

    # layer 2
    layer2_a = conv_layer(
        2, layer1_p, 64, is_training,
        filt=3, stride=1, pad='SAME',
    )

    layer2_p = tf.nn.max_pool(
        layer2_a,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='VALID'
    )

    # layer 3
    layer3_a = conv_layer(
        3, layer2_p, 128, is_training,
        filt=3, stride=1, pad='SAME',
    )

    layer3_p = tf.nn.max_pool(
        layer3_a,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='VALID'
    )

    print("Input: {}".format(images_ph.get_shape().as_list()))
    print("Layer 1: {}".format(layer1_p.get_shape().as_list()))
    print("Layer 2: {}".format(layer2_p.get_shape().as_list()))
    print("Layer 3: {}".format(layer3_p.get_shape().as_list()))

    #%%

    layer3_flat = tf.contrib.layers.flatten(layer3_p)

    layer4_a, _ = fc_layer(4, layer3_flat, 1152)

    layer4_a_drop = tf.nn.dropout(layer4_a, keep_prob_ph)

    _, layer5_z = fc_layer(5, layer4_a_drop, 10)

    layer_last_z = layer5_z

    print("Layer 4: {}".format(layer3_flat.get_shape().as_list()))
    print("Layer 5: {}".format(layer_last_z.get_shape().as_list()))

    #%% prediction evaluation nodes

    # argmax across which axis? last axis
    predicted_label = tf.argmax(layer_last_z, axis=1)
    true_label = tf.argmax(correct_labels_ph, axis=1)

    correct_prediction = tf.equal(predicted_label, true_label)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #%% compute loss

    loss_per_example = tf.nn.softmax_cross_entropy_with_logits(
        logits=layer_last_z,
        labels=correct_labels_ph,
    )

    cost = tf.reduce_mean(loss_per_example)

    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
    optimizer_step = optimizer.minimize(cost)

    #%%

    sess = tf.Session()

    saver = tf.train.Saver()

    results_final_model = pd.DataFrame(columns=[])

    # hyperparameter search
    exit_flag = False
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

                if True:
                    # https://stackoverflow.com/questions/25239933/how-to-add-title-to-subplots-in-matplotlib
                    # https://stackoverflow.com/questions/3823752/display-image-as-grayscale-using-matplotlib
                    # https://stackoverflow.com/questions/39659998/using-pyplot-to-create-grids-of-plots
                    fig = plt.figure()
                    ax1 = fig.add_subplot(1, 2, 1)
                    ax1.imshow(train_batches[0][0][0, :, :, 0], cmap='gray')
                    ax1.set_title("train [0]")
                    ax2 = fig.add_subplot(1, 2, 2)
                    ax2.imshow(dataset.dev_images[0, :, :, 0], cmap='gray')
                    ax2.set_title("dev [0]")
                    plt.show()
                    plt.close(fig)

                number_of_iterations = len(train_batches)
                cost_train_epoch_list = list()
                accuracy_train_epoch_list = list()
                for idx, (examples_images, examples_labels) in enumerate(train_batches):
                    cost_train_batch, accuracy_train_batch, _ = sess.run(
                        [cost, accuracy, optimizer_step], feed_dict={
                            images_ph: examples_images,
                            correct_labels_ph: examples_labels,
                            keep_prob_ph: keep_prob_val,
                            is_training: True,
                        }
                    )
                    cost_train_epoch_list.append(cost_train_batch)
                    accuracy_train_epoch_list.append(accuracy_train_batch)
                    cost_train_epoch = np.mean(cost_train_epoch_list)
                    accuracy_train_epoch = np.mean(accuracy_train_epoch_list)

                    if (idx + 1) % 10 == 0:
                        IPython.display.clear_output(wait=True)
                        print("epoch {:3d}, minibatch {:4d}/{:4d}, cost {:5.4f}, acc {:6.4f}%".format(
                            epoch, idx + 1, number_of_iterations,
                            cost_train_epoch,
                            accuracy_train_epoch * 100,
                        ), end="\r")

                # compute only entropy part of cost, without regularization
                cost_dev_epoch, accuracy_dev_epoch = sess.run([cost, accuracy], feed_dict={
                    images_ph: dataset.dev_images,
                    correct_labels_ph: dataset.dev_labels,
                    keep_prob_ph: 1.0,
                    is_training: False,
                })

                costs_run_table = costs_run_table.append({
                    'epoch': epoch,
                    'cost_train': cost_train_epoch,
                    'accuracy_train': accuracy_train_epoch * 100,
                    'cost_dev': cost_dev_epoch,
                    'accuracy_dev': accuracy_dev_epoch * 100,
                }, ignore_index=True)

                print(
                    "\nepoch {}; cost train {:5.4f}, dev {:5.4f}; acc train {:6.4f}%, dev {:6.4f}%".format(
                        epoch,
                        cost_train_epoch,
                        cost_dev_epoch,
                        accuracy_train_epoch * 100,
                        accuracy_dev_epoch * 100
                    )
                )

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
            sess, "tmp/model_{:02d}_{:2d}.ckpt".format(0, idx_keep_prob))
        print("Model saved in file: {}".format(save_path))

        results_final_model = results_final_model.append({
            'model': save_path,
            'keep_prob': keep_prob_val,
            'cost_train': cost_train_epoch,
            'accuracy_train': accuracy_train_epoch,
            'cost_dev': cost_dev_epoch,
            'accuracy_dev': accuracy_dev_epoch,
        }, ignore_index=True)

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
    test_images = DataContainer.array_to_img(dataset_test.data)

    predictions_test = sess.run(predicted_label, feed_dict={
        images_ph: test_images,
        keep_prob_ph: 1.0,
        is_training: False,
    })

    predictions_test_df = pd.DataFrame(predictions_test).reset_index()
    predictions_test_df.columns = ['ImageId', 'Label']
    predictions_test_df['ImageId'] += 1

    for idx, row in predictions_test_df.head(10).iterrows():
        fig = plt.figure()
        plt.imshow(test_images[idx, :, :, 0], cmap='gray')
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
