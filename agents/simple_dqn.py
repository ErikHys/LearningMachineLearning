from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import categorical_crossentropy

import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.input_shape = input_shape
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape[0], input_shape[1], input_shape[2]))
        self.new_state_memory = np.zeros((self.mem_size, input_shape[0], input_shape[1], input_shape[2]))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.mem_cntr = 0

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    model = Sequential([
        Dense(fc1_dims, input_shape=(input_dims, ), activation="relu"),
        Dense(fc2_dims, activation="relu"),
        Dense(n_actions)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    return model


def build_cnn_dqn(lr, n_actions, input_dims):
    model = Sequential([
        Conv2D(32, (6, 6), activation='relu', kernel_initializer='he_uniform', input_shape=input_dims),
        MaxPool2D((4, 4)),
        Flatten(),
        Dense(n_actions, activation='softmax')
    ])

    model.compile(optimizer=SGD(learning_rate=lr, momentum=0.9),
                  loss=categorical_crossentropy,
                  metrics=['accuracy'])
    return model


class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=0.996, epsilon_end=0.01,
                 mem_size=1000000, fname='dqn_model.h5', cnn=False):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        if cnn:
            self.q_eval = build_cnn_dqn(alpha, n_actions, input_dims)
        else:
            self.q_eval = build_dqn(alpha, n_actions, input_dims, 32, 32)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)
        return action

    def choose_car_action(self, state):
        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.array([np.random.random(1), np.random.random(1), np.random.random(1)]).flatten()
        else:
            actions = self.q_eval.predict(state)
            action = actions
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        actions_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, actions_values)

        q_eval = self.q_eval.predict(state)
        q_next = self.q_eval.predict(new_state)

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = reward + self.gamma*np.max(q_next, axis=1)*done

        _ = self.q_eval.fit(state, q_target, verbose=0)

        self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)
