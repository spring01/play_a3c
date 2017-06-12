import tensorflow as tf
import sys
import numpy as np


# Build model...
inp = tf.placeholder(tf.float32, shape=(None, 100))
weights = tf.Variable(tf.random_normal([100, 50], stddev=0.35))
out = tf.matmul(inp, weights)
target = tf.placeholder(tf.float32, shape=(None, 50))
loss = tf.nn.l2_loss(target - out)
train_op = tf.train.AdamOptimizer(0.0).minimize(loss)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
      inter_op_parallelism_threads=1)
sess = tf.Session(config=session_conf)
init = tf.global_variables_initializer()
sess.run(init)
inp_batch = np.random.rand(320, 100)
out_batch = np.random.rand(320, 50)
while True:
    result = sess.run(train_op, feed_dict={inp: inp_batch, target: out_batch})


