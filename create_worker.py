import sys
task_number = int(sys.argv[1])

import tensorflow as tf

cluster_list = ['localhost:222%d' % num for num in range(8)]
cluster = tf.train.ClusterSpec({'local': cluster_list})
server = tf.train.Server(cluster, job_name="local", task_index=task_number)

print("Starting server #{}".format(task_number))

server.start()
server.join()
