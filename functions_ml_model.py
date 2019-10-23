# -*- coding: utf-8 -*-
"""
@author: huongha
@description: Some common machine learning models for testing Bayesian optimization
"""

import numpy as np
import math

from collections import OrderedDict
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

import tensorflow as tf


def reshape(x, input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size == input_dim:
        x = x.reshape((1, input_dim))
    return x


def random_mini_batches(X, Y, batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (number of examples, input size)
    Y -- true "label" vector of shape (number of examples, number of features)
    batch_size -- size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random
    minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :].reshape((m, Y.shape[1]))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_minibatches = math.floor(m/batch_size)
    for k in range(0, num_minibatches):
        mini_batch_X = shuffled_X[k*batch_size:k*batch_size + batch_size, :]
        mini_batch_Y = shuffled_Y[k*batch_size:k*batch_size + batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < batch_size)
    if m % batch_size != 0:
        mini_batch_X = shuffled_X[num_minibatches*batch_size:m, :]
        mini_batch_Y = shuffled_Y[num_minibatches*batch_size:m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


class functions:
    def plot(self):
        print("Not implemented")


class Elastic_net:

    '''
    ElaticNet_function: function

    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, bounds=None, sd=None):
        self.input_dim = 2

        if bounds is None:
            self.bounds = OrderedDict([('alpha', (-100, 100)),
                                       ('l1_ratio', (1e-8, 1))])
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = 1
        self.ismax = 1
        self.name = 'Elastic_net'

    def func(self, params, X_train, Y_train, X_test, Y_test):
        # params: list of hyperparameters need to optimize

        if (type(params) == list):
            metrics_accuracy = np.zeros((len(params), 1))
            for i in range(len(params)):
                alpha = 10**params[i][0]
                l1_ratio = params[i][1]

                clf = SGDClassifier(loss="log", penalty="elasticnet",
                                    alpha=alpha, l1_ratio=l1_ratio,
                                    max_iter=20, shuffle=True, random_state=1)
                clf.fit(X_train, Y_train)
                Y_test_predict = clf.predict(X_test)
                accuracy_temp = accuracy_score(Y_test, Y_test_predict)
                metrics_accuracy[i, 0] = accuracy_temp
        elif (type(params) == np.ndarray):
            alpha = 10**params[0]
            l1_ratio = params[1]

            clf = SGDClassifier(loss="log", penalty="elasticnet",
                                alpha=alpha, l1_ratio=l1_ratio,
                                max_iter=20, shuffle=True, random_state=1)
            clf.fit(X_train, Y_train)
            Y_test_predict = clf.predict(X_test)
            metrics_accuracy = accuracy_score(Y_test, Y_test_predict)
        else:
            print('Something wrong with params!')

        return metrics_accuracy


class DeepLearning_CNN_MNIST_TF:

    '''
    CNN_MNIST_function: function

    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, bounds=None, sd=None, seed=0):
        self.input_dim = 3

        # Oly tune 2 dropout params, learning rate, decay, momentum (of SGD)
        if bounds is None:
            self.bounds = OrderedDict([('dropout1', (1e-8, 1)),
                                       ('dropout2', (1e-8, 1)),
                                       ('lr', (-100, 100))])
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = 1
        self.ismax = 1
        self.name = 'DeepLearning_CNN_MNIST_TF'
        self.seed = seed
        tf.reset_default_graph()  # Safeguard just in case
        tf.set_random_seed(self.seed)  # Set seed for tensorflow
        np.random.seed(self.seed)      # Set seed for numpy

    def run_CNN(self, params, X_train, Y_train, X_test, Y_test):
        # NOTE: params has len being 1

        # Reset or set seed
        tf.reset_default_graph()  # Safeguard just in case
        tf.set_random_seed(0)  # Set seed for tensorflow
        np.random.seed(0)      # Set seed for numpy

        # Define some fixed hyperparameters
        num_classes = 10
        num_epochs = 20

        # Extract hyperparameters from params:
        dropout_rate1 = params[0]
        dropout_rate2 = params[1]
        learning_rate = 10**(params[2])
        batch_size = 128

        # Compute sample size of training and testing dataset
        train_sample_size = X_train.shape[0]
        test_sample_size = X_test.shape[0]

        # Define placeholder
        # Reshape to (batch, height, width, channel)
        tf_x = tf.placeholder(tf.float32, [None, 28*28])
        image = tf.reshape(tf_x, [-1, 28, 28, 1])
        tf_y = tf.placeholder(tf.int32, [None, 10])

        # First convo layer
        conv1 = tf.layers.conv2d(inputs=image,  # shape(28,28,1))
                                 filters=32,
                                 kernel_size=5,
                                 strides=1,
                                 padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1))  # -> shape(28,28,32))
        pool1 = tf.layers.max_pooling2d(conv1,
                                        pool_size=2,
                                        strides=2)  # -> shape(14, 14, 32)
        pool1 = tf.nn.dropout(pool1, dropout_rate1)
        
        # Second convo layer
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64,
                                 kernel_size=5,
                                 strides=1,
                                 padding='same',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1))  # -> shape (14, 14, 64)
        pool2 = tf.layers.max_pooling2d(conv2,
                                        pool_size=2,
                                        strides=2)  # -> shape (7, 7, 64)
        pool2 = tf.nn.dropout(pool2, dropout_rate2)

        # Output
        flat = tf.reshape(pool2, [-1, 7*7*64])  # -> shape (7*7*64, )
        output = tf.layers.dense(flat, num_classes,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1))

        # Loss function and training operation
        loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y,
                                               logits=output)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1),
                                       predictions=tf.argmax(output, axis=1),)[1]

        sess = tf.Session()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        # Training model
#        seed = self.seed
        seed = 0
        for epoch in range(num_epochs):

            epoch_loss = 0
            num_batches = int(train_sample_size/batch_size)
            seed = seed + 1
            batches = random_mini_batches(X_train,
                                          Y_train,
                                          batch_size=batch_size,
                                          seed=seed)
            for step, batch in enumerate(batches):

                # Select a minibatch
                b_X, b_Y = batch

                # Run the session to execute the optimizer and the cost
                _, b_loss = sess.run([train_op, loss], {tf_x: b_X, tf_y: b_Y})

                epoch_loss += b_loss/num_batches

#                if step % 200 == 0:
#                    accuracy_, flat_rep = sess.run([accuracy, flat],
#                                                   {tf_x: X_test, tf_y: Y_test})
#                    print('Step:', step, '| train loss: %.4f' % b_loss,
#                          '| test accuracy: %.2f' % accuracy_)

            if epoch % 5 == 0:
                accuracy_, flat_rep = sess.run([accuracy, flat],
                                               {tf_x: X_test, tf_y: Y_test})
                print("Cost after epoch %i: %f | accuracy %f" % (epoch,
                                                                 epoch_loss,
                                                                 accuracy_))

        accuracy_test, flat_test = sess.run([accuracy, flat],
                                            {tf_x: X_test, tf_y: Y_test})

        return accuracy_test

    def func(self, params, X_train, Y_train, X_test, Y_test):

        if (type(params) == list):
            metrics_accuracy = np.zeros((len(params), 1))
            for i in range(len(params)):
                params_single = params[i]
                accuracy_temp = self.run_CNN(params_single, X_train,
                                             Y_train, X_test, Y_test)
                metrics_accuracy[i, 0] = accuracy_temp

        elif (type(params) == np.ndarray):
            # import os
            # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
            params_single = params.copy()
            metrics_accuracy = self.run_CNN(params_single, X_train,
                                            Y_train, X_test, Y_test)
        else:
            print('Something wrong with params!')

        return metrics_accuracy


class MLP_MNIST_TF:
    '''
    MLP_MNIST_function: function

    param sd: standard deviation, to generate noisy evaluations of the function
    '''
    def __init__(self, bounds=None, sd=None, seed=0):
        self.input_dim = 3

        # Oly tune 2 dropout params, learning rate, decay, momentum (of SGD)
        if bounds is None:
            self.bounds = OrderedDict([('l1_reg', (-100, 100)),
                                       ('l2_reg', (-100, 100)),
                                       ('lr', (-100, 100))])
        else:
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = 1
        self.ismax = 1
        self.name = 'DeepLearning_MLP_MNIST_TF'
        self.seed = seed
        tf.reset_default_graph()  # Safeguard just in case
        tf.set_random_seed(self.seed)  # Set seed for tensorflow
        np.random.seed(self.seed)      # Set seed for numpy

    def run_MLP(self, params, X_train, Y_train, X_test, Y_test):
        # NOTE: params has len being 1

        # Reset or set seed
        tf.reset_default_graph()  # Safeguard just in case
        tf.set_random_seed(0)  # Set seed for tensorflow
        np.random.seed(0)      # Set seed for numpy

        # Define some fixed hyperparameters
        num_classes = 10
        num_epochs = 20
        num_neurons = 512

        # Extract hyperparameters from params:
        l1_reg = 10**params[0]
        l2_reg = 10**params[1]
        learning_rate = 10**(params[2])
        batch_size = 128

        # Compute sample size of training and testing dataset
        train_sample_size = X_train.shape[0]

        # Define placeholder
        # Reshape to (batch, height, width, channel)
        tf_x = tf.placeholder(tf.float32, [None, 28*28])
        tf_y = tf.placeholder(tf.int32, [None, num_classes])

        # First layer
        layer_1 = tf_x
        layer_1 = tf.layers.dense(layer_1, num_neurons, tf.nn.relu,
                                  kernel_initializer = tf.contrib.layers.xavier_initializer(seed = 1),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(float(l1_reg)))

        # Second layer
        layer_2 = tf.layers.dense(layer_1, num_neurons, tf.nn.relu,
                                  kernel_initializer = tf.contrib.layers.xavier_initializer(seed = 1),
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(float(l2_reg)))
        output = tf.layers.dense(layer_2, 10)

        # Define the loss function
        l2_loss = tf.losses.get_regularization_loss()
        loss = l2_loss + tf.losses.mean_squared_error(output, tf_y)

        # Define the training operation
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1),
                                       predictions=tf.argmax(output, axis=1),)[1]

        sess = tf.Session()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        # Training model
        seed = 0
        for epoch in range(num_epochs):

            epoch_loss = 0
            num_batches = int(train_sample_size/batch_size)
            seed = seed + 1
            batches = random_mini_batches(X_train,
                                          Y_train,
                                          batch_size=batch_size,
                                          seed=seed)
            for step, batch in enumerate(batches):

                # Select a minibatch
                b_X, b_Y = batch

                # Run the session to execute the optimizer and the cost
                _, b_loss = sess.run([train_op, loss], {tf_x: b_X, tf_y: b_Y})

                epoch_loss += b_loss/num_batches

            if epoch % 5 == 0:
                accuracy_ = sess.run(accuracy,
                                     {tf_x: X_test, tf_y: Y_test})
                print("Cost after epoch %i: %f | accuracy %f" % (epoch,
                                                                 epoch_loss,
                                                                 accuracy_))

        accuracy_test = sess.run(accuracy, {tf_x: X_test, tf_y: Y_test})

        return accuracy_test

    def func(self, params, X_train, Y_train, X_test, Y_test):

        if (type(params) == list):
            metrics_accuracy = np.zeros((len(params), 1))
            for i in range(len(params)):
                params_single = params[i]
                accuracy_temp = self.run_MLP(params_single, X_train,
                                             Y_train, X_test, Y_test)
                metrics_accuracy[i, 0] = accuracy_temp

        elif (type(params) == np.ndarray):
            # import os
            # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
            params_single = params.copy()
            metrics_accuracy = self.run_MLP(params_single, X_train,
                                            Y_train, X_test, Y_test)
        else:
            print('Something wrong with params!')

        return metrics_accuracy

