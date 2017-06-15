
import sys
import pickle
import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as keras
from ppss import PPSSWrapper

def main():
    env = gym.make('Breakout-v0')
    env = PPSSWrapper(env)

    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

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
    model = keras.models.Model(inputs=input_h, outputs=[value, policy_logits])


    state_h = model.inputs[0]
    value, policy_logits = model.outputs
    policy_action = tf.squeeze(tf.multinomial(policy_logits - tf.reduce_max(policy_logits, [1], keep_dims=True), 1), [1])
    policy_action = tf.one_hot(policy_action, num_actions)

    sess = tf.Session()
    init_global = tf.global_variables_initializer()
    sess.run(init_global)

    with open(sys.argv[1], 'rb') as save:
        weights = pickle.load(save)

    model_weights = model.weights
    assignments = tf.group(*[mw.assign(w) for mw, w in zip(model_weights, weights)])
    sess.run(assignments)

    print(weights[-2])
    print(sess.run(model.weights[-2]))
    all_total_rewards = []
    for _ in range(20):
        state = env.reset()
        env.unwrapped.render()
        total_rewards = 0.0
        for i in range(100000):
            action_onehot = sess.run(policy_action, feed_dict={state_h: [state]})[0]
            #~ print(action_onehot)
            state, reward, done, info = env.step(action_onehot.argmax())
            env.unwrapped.render()
            total_rewards += reward
            if done:
                break
        all_total_rewards.append(total_rewards)
        print('episode reward:', total_rewards)
    print('average episode reward:', np.mean(all_total_rewards))


if __name__ == "__main__":
    main()
