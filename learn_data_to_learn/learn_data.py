'''
Implementation of the paper-----Learning What Data to Learn.
Author: Linzhuo Pang
Project: https://github.com/plz717/learning_to_learn
'''

import matplotlib
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import sys
import numpy as np
import tensorflow as tf

import collections
import pickle
from sklearn.utils import shuffle
from tool import image_dataset
import heapq

# partameters of policy learning
L = 1000
T = 100
M = 30  # mini_batchsize
validtion_iterations = 30
accuracy_threshold = 0.8
gamma = 0.93
episode_threshold = 100

# parameters of using the policy to select data and train the classifier
MAX_ITER = 3000


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name)


def bias_variable(shape, name):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name)


def shape(tensor):
    '''
        return the shape of a tensor in tuple format
    '''
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])


def BN(input_tensor, axis):
    depth = shape(input_tensor)[-1]
    beta = tf.Variable(tf.constant(
        0.0, shape=[depth]), name='beta', trainable=True)
    gama = tf.Variable(tf.constant(
        1.0, shape=[depth]), name='gama', trainable=True)
    batch_mean, batch_variance = tf.nn.moments(input_tensor, axis)
    normalized_tensor = tf.nn.batch_normalization(
        input_tensor, batch_mean, batch_variance, beta, gama, variance_epsilon=1e-3)
    return normalized_tensor


class Classifier():

    def __init__(self, learning_rate=0.0001, scope="Classifier", add_BN=True):

        self.x = tf.placeholder("float", [None, 784])
        self.y_ = tf.placeholder("float", [None, 10])

        W1 = weight_variable([784, 300], "W1")
        b1 = weight_variable([300], "b1")
        W2 = weight_variable([300, 10], "W2")
        b2 = weight_variable([10], "b2")

        layer_1 = tf.matmul(self.x, W1) + b1
        fc_1 = tf.nn.sigmoid(
            BN(layer_1, axis=[0])) if add_BN == True else tf.nn.sigmoid(layer_1)

        layer_2 = tf.matmul(fc_1, W2) + b2
        fc_2 = tf.nn.sigmoid(
            BN(layer_2, axis=[0])) if add_BN == True else tf.nn.sigmoid(layer_2)

        y = fc_2
        self.prediction = y
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y))
        self.train_step = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(self.cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def update(self, x, y_, sess):
        sess = sess or tf.get_default_session()
        sess.run(self.train_step, feed_dict={self.x: x, self.y_: y_})

    def predict(self, x, sess):
        sess = sess or tf.get_default_session()
        return sess.run(self.prediction, feed_dict={self.x: x})

    def get_loss(self, x, y_, sess):
        sess = sess or tf.get_default_session()
        return sess.run(self.cross_entropy, feed_dict={self.x: x, self.y_: y_})

    def get_accuracy(self, x, y_, sess):
        sess = sess or tf.get_default_session()
        return sess.run(self.accuracy, feed_dict={self.x: x, self.y_: y_})


def state_preprocess(classifier, M, train_x, train_y, training_loss_list, iterations):
    """This function is to generate 24-dstates for actor.
    states is comprised of three parts:
    1.data features:label of the input data   (10-d)
    2.model features:passed mini-batch number(i.e iterations),  1-d
                     average training loss of the classifier, 1-d
                     validation acuracy of the model   1-d
    3.relationships of data and model:predicted label for the input data  10-d
                                      cross entropy on the input data   1-d
    """
    cross_entropy = classifier.get_loss(
        np.array(train_x), np.array(train_y), sess)
    training_loss_list.append(cross_entropy)
    average_loss = np.mean(np.array(training_loss_list))
    validation_accuracy = classifier.get_accuracy(
        validation_x, validation_y, sess)

    # data features
    true_label = train_y  # (M,10)
    # model features
    iterations_array = np.ones((M, 1)) * iterations  # (M,1)
    average_loss_array = np.ones((M, 1)) * average_loss  # (M,1)
    validation_accuracy_array = np.ones((M, 1)) * validation_accuracy  # (M,1)
    # both features
    cross_entropy_array = np.ones((M, 1)) * cross_entropy  # (M,1)
    predict_label = classifier.predict(
        train_x, sess=sess)  # (M,10)

    processed_state = np.concatenate([true_label, iterations_array, average_loss_array,
                                      validation_accuracy_array, cross_entropy_array, predict_label], axis=1)  # (M,24)
    print("processded state shape is:".format(processed_state.shape))
    return processed_state


def get_reward(classifier, validation_x, validation_y, validtion_iterations, accuracy_threshold, sess=None):
    """
    calculate the reward r_T when t=T (termianl)
    """
    sess = sess or tf.get_default_session()
    for i in range(validtion_iterations):
        validation_X, validation_Y = shuffle(
            validation_x, validation_y)
        validation_set = image_dataset(validation_X, validation_Y)
        valid_x, valid_y = validation_set.next_batch(M)
        valid_accuracy = classifier.get_accuracy(valid_x, valid_y, sess)
        if valid_accuracy > accuracy_threshold:
            r_T = -tf.log(i / T)
            return r_T
    return 0


class PolicyEstimator():
    """objective function is: log pi(a|s) * (r_t - baseline)
    the network_structure is: 24*12*1
    """

    def __init__(self, learning_rate=1e-6, scope="policy estimator", add_BN=True):

        self.state = tf.placeholder("float", [None, 24])
        self.action = tf.placeholder("float", [None, 1])
        self.target = tf.placeholder(dtype=tf.float32, name="target")

        W1 = weight_variable([24, 12], "W1")
        b1 = weight_variable([12], "b1")
        W2 = weight_variable([12, 1], "W2")
        b2 = weight_variable([1], "b2")

        layer_1 = tf.matmul(self.state, W1) + b1
        fc_1 = tf.nn.sigmoid(
            BN(layer_1, axis=[0])) if add_BN == True else tf.nn.sigmoid(layer_1)

        layer_2 = tf.matmul(fc_1, W2) + b2
        fc_2 = tf.nn.sigmoid(
            BN(layer_2, axis=[0])) if add_BN == True else tf.nn.sigmoid(layer_2)

        self.action_probs = fc_2  # batchsize*1
        self.picked_action_prob = tf.reduce_sum(
            tf.multiply(self.action_probs, self.action), reduction_indices=1)  # (32,)
        self.target = tf.cast(self.target, tf.float32)

        self.loss = - \
            tf.reduce_sum(tf.log(self.picked_action_prob) * self.target)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(
            self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        # , self.action_probs
        print(sess.run([self.action_probs], feed_dict={self.state: state}))
        return sess.run([self.action_probs], feed_dict={self.state: state})

    def update(self, state, action, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state,
                     self.action: action, self.target: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss


def NDF_REINFORCE(actor, classifier, train_x, train_y, validation_x, validation_y, M, dis_factor=gamma, episode_num=L, sess=None):
    sess = sess or tf.get_default_session()
    Transition = collections.namedtuple(
        "Transition", ["state", "action", "reward"])
    training_loss_list = []
    for l in range(1, L + 1):
        episode = []

        # shuffle D' to get minibatches sequence D'={D1,D2,...,DT}
        train_X, train_Y = shuffle(train_x, train_y)
        trainset = image_dataset(train_X, train_Y)
        r_t_list = []
        for t in range(1, T + 1):
            train_x, train_y = trainset.next_batch(M)
            processed_state = state_preprocess(
                classifier, M, train_x, train_y, training_loss_list, iterations=l * T)
            action_probs = actor.predict(processed_state, sess)  # (1,M,1)
            list_action_probs = list(action_probs[0].reshape(M))  # (M,)
            print("list_action_probs is:{}".format(list_action_probs))
            # project to 1
            if l < episode_threshold:
                action_nlargest = heapq.nlargest(
                    int(M / 2), list_action_probs)
                print("action_nlargest is:{}".format(action_nlargest))
                action_flag = [
                    1 if item in action_nlargest else 0 for item in list_action_probs]
                print("l < episode_threshold, and  action flag is:{}".format(
                    action_flag))

            else:
                action_flag = [1 if item >
                               0.5 else 0 for item in list_action_probs]
                print("l >= episode_threshold, and action flag is:{}".format(
                    action_flag))
            # store the selected train samples
            select_train_x = []
            select_train_y = []
            for index, item in enumerate(action_flag):
                if item == 1:
                    select_train_x.append(train_x[index])
                    select_train_y.append(train_y[index])
            # select_train_x = [train_x[index] if item == 1 for index, item in
            # enumerate(action_flag)]
            print("selected train_samples number is:{}".format(
                len(select_train_x)))
            print("select_train_x shape is:{}".format(
                np.array(select_train_x).shape))
            # update the classifier using selected samples
            if len(select_train_x) != 0:
                classifier.update(np.array(select_train_x),
                                  np.array(select_train_y), sess)

            # only when terminal receives reward
            if t == T:
                r_t = get_reward(classifier, validation_x, validation_y,
                                 validtion_iterations, accuracy_threshold, sess=sess)
            else:
                r_t = 0
            r_t_list.append(r_t)

            episode.append(Transition(state=processed_state, action=np.array(
                action_flag).reshape(-1, 1), reward=r_t))  # (-1,24),(-1,1)
        print("r_t_list is:{}".format(r_t_list))

        for t, trainsition in enumerate(episode):
            v_t = sum(dis_factor**i * item.reward for i,
                      item in enumerate(episode[t:]))
            print("v_t is:{}".format(v_t))
            actor.update(trainsition.state,
                         trainsition.action, v_t, sess=sess)


def train_using_policy(input_x, input_y, classifier, policyestimator, sess=None):
    sess = sess or tf.get_default_session()
    test_x = mnist.test.images
    test_y = mnist.test.labels
    test = image_dataset(test_x, test_y)
    train = image_dataset(input_x, input_y)

    for i in range(MAX_ITER):
        x, y = train.next_batch(M)
        action_probs = policyestimator.predict(x, sess)  # (M,1)
        list_action_probs = [item[0] for item in list(action_probs)]
        # project to 1
        action_flag = [1 if item >
                       0.5 else 0 for item in list_action_probs]
        # store the selected train samples
        select_train_x = []
        select_train_y = []
        for index, item in enumerate(action_flag):
            if item == 1:
                select_train_x.append(x[index])
                select_train_y.append(y[index])
        print("selected train_samples number is:{}".format(len(select_train_x)))
        # update the classifier using selected samples
        classifier.update(np.array(select_train_x),
                          np.array(select_train_y), sess)
        train_loss = classifier.get_loss(np.array(select_train_x),
                                         np.array(select_train_y), sess=sess)
        test_accuracy = classifier.get_accuracy(test.next_batch(
            M)[0], test.next_batch(M)[1], sess=sess)
        print("during training, the train loss is:{}, test_accuracy is:{}".format(
            train_loss, test_accuracy))


if __name__ == '__main__':

    '''
    # for test PolicyEstimator
    x = PolicyEstimator()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        state = np.ones((10, 24), dtype=np.float32)
        state = state * 10
        state[:5, :] = 3
        print(x.predict(state))
    '''
    '''
    # for test Classifier
    classifier = Classifier()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        x = train_x[:10, :]
        label = train_y[:10, :]
        print(classifier.get_loss(x, label, sess))
    '''
    with open('../flipped_train_x', 'rb') as f:
        splited_train_x = pickle.load(f)
    with open('../flipped_train_y', 'rb') as f:
        splited_train_y = pickle.load(f)

    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

    splited_train_x = np.array(splited_train_x).reshape(55000, 784)
    splited_train_y = np.array(splited_train_y).reshape(55000, 10)

    train_x = splited_train_x[:10000, :]  # train the policy learner
    train_y = splited_train_y[:10000, :]
    # validation set used for reward calculation
    validation_x = splited_train_x[45000:, :]
    validation_y = splited_train_y[45000:, :]

    classifier = Classifier()
    actor = PolicyEstimator()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        NDF_REINFORCE(actor, classifier, train_x, train_y, validation_x, validation_y, M,
                      episode_num=L, dis_factor=gamma, sess=sess)

    x = splited_train_x[10000:45000, :]
    y = splited_train_y[10000:45000, :]

    train_using_policy(x, y, classifier, actor, sess=None)
