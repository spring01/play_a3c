
import argparse
import gym
from wrapper import *
from network.qnetwork import *

env = gym.make('Breakout-v0')
wenv = HistoryWrapper(PreprocessWrapper(env))
wenv.reset()
wenv.render(mode='wrapped')
for _ in range(100):
    state, _, _, _ = wenv.step(wenv.action_space.sample())
    wenv.render(mode='wrapped')

#~ env2 = gym.make('CartPole-v0')
#~ wenv2 = HistoryWrapper(env2)
#~ wenv2.reset()
#~ wenv2.render()
#~ for _ in range(5):
    #~ state, _, _, _ = wenv2.step(wenv2.action_space.sample())
    #~ wenv2.render()


#~ input_shape = wenv.observation_space.shape
#~ num_actions = wenv.action_space.n

#~ parser = argparse.ArgumentParser(description='Play game with DQN')
#~ qnetwork_add_arguments(parser)
#~ args = parser.parse_args()

#~ qnet = qnetwork(input_shape, num_actions, args)
#~ qnet.summary()
