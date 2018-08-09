# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 15:52:26 2018

@author: lhe39759
"""
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def buildModel():
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    
    
    x = tf.placeholder(tf.float32, [None, 784])
    
    ####################
    #Dense
    ####################
    W_1 = weight_variable([784, 4])
    b_1= bias_variable([4])
    
    h_1 = tf.nn.tanh(tf.matmul(x, W_1) + b_1)
    
    W_2 = weight_variable([4, 3])
    b_2= bias_variable([3])
    
    h_2 = tf.nn.tanh(tf.matmul(h_1, W_2) + b_2)
    
    
    ###################
    #Output
    ####################
    W_Out = weight_variable([3, 10])
    b_Out = bias_variable([10])
    
    y_conv = tf.matmul(h_2, W_Out) + b_Out
    
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    return cross_entropy, train_step, correct_prediction, accuracy
  
#Loads MNIST Data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Connection to tf backend
sess = tf.InteractiveSession()

cross_entropy, train_step, correct_prediction, accuracy = buildModel()

sess.run(tf.global_variables_initializer())

for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))