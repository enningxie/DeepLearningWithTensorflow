# coding=utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

# 通过tf.device将运算指定到特定的设备上
with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
    b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
with tf.device('/gpu:0'):
    c = a + b

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))