
import sys
import pickle
import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as keras
from ppss import PPSSWrapper


def main():
    num_workers = int(sys.argv[2])
    port_list = [2220 + num for num in range(num_workers)]
    task_index = int(sys.argv[1])
    port = port_list[task_index]

    cluster_list = ['localhost:{}'.format(port) for port in port_list]
    cluster = tf.train.ClusterSpec({'local': cluster_list})
    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    server = tf.train.Server(cluster, job_name='local', task_index=task_index,
                             config=config)
    print('Starting server #{}'.format(task_index))


    env = gym.make('Breakout-v0')
    env = PPSSWrapper(env)

    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    worker_dev = '/job:local/task:{}/cpu:0'.format(task_index)
    rep_dev = tf.train.replica_device_setter(worker_device=worker_dev,
                                             cluster=cluster)

    with tf.device(rep_dev):
        input_h = keras.layers.Input(shape=input_shape)
        Conv2D = keras.layers.Conv2D
        conv1_32 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        conv2_64 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        conv3_64 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')
        conv1 = conv1_32(input_h)
        conv2 = conv2_64(conv1)
        convf = conv3_64(conv2)
        feature = keras.layers.Flatten()(convf)
        Dense = keras.layers.Dense
        hid = Dense(512, activation='relu')(feature)
        value = Dense(1)(hid)
        policy_logits = Dense(num_actions)(hid)
        model_global = keras.models.Model(inputs=input_h, outputs=[value, policy_logits])
        step_global = tf.Variable(0)

    with tf.device(worker_dev):
        global_weights = model_global.weights
        value, policy_logits = model_global.outputs
        value = value[:, 0]
        log_policy_prob = tf.nn.log_softmax(policy_logits)
        policy_prob = tf.nn.softmax(policy_logits)

        policy_action = tf.squeeze(tf.multinomial(policy_logits - tf.reduce_max(policy_logits, [1], keep_dims=True), 1), [1])
        policy_action = tf.one_hot(policy_action, num_actions)

        state_h = model_global.inputs[0]
        advantage_h = tf.placeholder(tf.float32, [None])
        reward_h = tf.placeholder(tf.float32, [None])
        action_h = tf.placeholder(tf.float32, [None, num_actions])

        policy_loss = -tf.reduce_sum(tf.reduce_sum(log_policy_prob * action_h, [1]) * advantage_h)
        value_loss = 0.5 * tf.reduce_sum(tf.square(value - reward_h))
        entropy = -tf.reduce_sum(policy_prob * log_policy_prob)
        loss = policy_loss + 0.5 * value_loss - entropy * 0.01

        opt = tf.train.AdamOptimizer(1e-4)
        grads = tf.gradients(loss, global_weights)
        grads, _ = tf.clip_by_global_norm(grads, 40.0)
        grads_and_vars = list(zip(grads, global_weights))

        inc_step = step_global.assign_add(tf.shape(state_h)[0])
        train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)

    with tf.Session('grpc://localhost:{}'.format(port)) as sess:
        init_global = tf.global_variables_initializer()
        sess.run(init_global)

        step = sess.run(step_global)
        with open('save{}.p'.format(step), 'wb') as pic:
            pickle.dump(sess.run(global_weights), pic)

        state = env.reset()
        last_step = sess.run(step_global)
        while True:
            state_history = [state]
            action_history = []
            reward_history = []
            for t in range(5):
                action_onehot = sess.run(policy_action, feed_dict={state_h: [state]})[0]
                #~ print(action_onehot)
                state, reward, done, info = env.step(action_onehot.argmax())
                if reward > 0.0:
                    reward = 1.0
                elif reward < 0.0:
                    reward = -1.0
                state_history.append(state)
                action_history.append(action_onehot)
                reward_history.append(reward)
                if done:
                    state = env.reset()
                    break
            batch_state = np.stack(state_history)
            batch_action = np.stack(action_history)
            batch_value = sess.run(value, feed_dict={state_h: batch_state})
            reward_long = 0.0 if done else batch_value[-1]
            reward_long_list = []
            for reward in reversed(reward_history):
                reward_long = reward + 0.99 * reward_long
                reward_long_list.append(reward_long)
            batch_reward = np.stack(reversed(reward_long_list))

            batch_adv = batch_reward - batch_value[:-1]
            result = sess.run(train_op, feed_dict={state_h: batch_state[:-1],
                                                   advantage_h: batch_adv,
                                                   reward_h: batch_reward,
                                                   action_h: batch_action})
            #~ print(sess.run(global_weights[-2]))
            if task_index == 0:
                step = sess.run(step_global)
                if step - last_step > 10000:
                    save_weights = sess.run(global_weights)
                    with open('save{}.p'.format(step), 'wb') as pic:
                        pickle.dump(save_weights, pic)
                    print(save_weights[-2])
                    last_step = step
                print(step)


if __name__ == "__main__":
    main()
