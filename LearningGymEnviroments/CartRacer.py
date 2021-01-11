import gym
import keyboard

env = gym.make('CarRacing-v0')
observation = env.reset()

for i in range(1000):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()
env.close()
