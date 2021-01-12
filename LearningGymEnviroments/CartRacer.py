import gym
from utils import plotLearning
import numpy as np
import agents.simple_dqn


def run():
    env = gym.make('CarRacing-v0')
    n_games = 500
    agent = agents.simple_dqn.Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, input_dims=(96, 96, 3), n_actions=3,
                                    mem_size=10000, batch_size=64,
                                    epsilon_end=0.01, cnn=True, discrete=False)

    scores = []
    eps_history = []
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            action = agent.choose_car_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode ', i, 'score %.2f' % score, ' avg score %.2f' % avg_score)

    env.close()

    x = [i+1 for i in range(n_games)]

    plotLearning(x, scores, eps_history, 'lunarlanding.png')

run()