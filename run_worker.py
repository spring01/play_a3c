import tensorflow as tf
import sys
import numpy as np


cluster_list = ['localhost:222{:d}'.format(num) for num in range(int(sys.argv[2]))]
cluster = tf.train.ClusterSpec({'local': cluster_list})

task_number = int(sys.argv[1])
port = '222{:d}'.format(task_number)

config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1)
server = tf.train.Server(cluster, job_name='local', task_index=task_number, config=config)
print('Starting server #{}'.format(task_number))
#~ server.start()

worker_device = '/job:local/task:{}/cpu:0'.format(task_number)
rep_dev = tf.train.replica_device_setter(worker_device=worker_device, cluster=cluster)
with tf.device(rep_dev):
    # Build model...
    inp = tf.placeholder(tf.float32, shape=(None, 100))
    weights = tf.Variable(tf.random_normal([100, 50], stddev=0.35))
    out = tf.matmul(inp, weights)
    target = tf.placeholder(tf.float32, shape=(None, 50))
    loss = tf.nn.l2_loss(target - out)
    train_op = tf.train.AdamOptimizer(0.0).minimize(loss)


with tf.Session('grpc://localhost:{}'.format(port)) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    inp_batch = np.random.rand(320, 100)
    out_batch = np.random.rand(320, 50)
    while True:
        sess.run(out, feed_dict={inp: inp_batch})
        result = sess.run(train_op, feed_dict={inp: inp_batch, target: out_batch})
        print('***** at worker {} start *****'.format(task_number))
        print(sess.run(weights))
        print(inp_batch)
        print('***** at worker {} end *****'.format(task_number))
    print(result)
