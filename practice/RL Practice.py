# pip install gym==0.25.2 tensorflow keras-rl2 pygame==2.6.0 numpy==1.26.0
# go to environment's Lib/site-packages/rl/util.py and replace model_from_config with model_from_json 

import random
import gym

# import tensorflow as tf
# print(tf.__version__)
# from tensorflow.keras.models import Sequential


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make("CartPole-v1", render_mode="human")

states = env.observation_space.shape[0]
actions = env.action_space.n

# print(actions)
# print(states)



episodes = 10

for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = random.choice([0, 1])
        n_state, reward, done, info = env.step(action)
        score += reward
        env.render()

    print(f"Episode: {episode}  Score: {score}")

env.close()