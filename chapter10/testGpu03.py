# coding=utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


a_cpu = tf.Variable(0, name="a_cpu")
with tf.device('/gpu:0'):
    a_gpu = tf.Variable(0, name="a_gpu")
    # 通过allow_soft_placement参数自动将无法放在GPU上的操作放回CPU上
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess.run(tf.global_variables_initializer())