import sys
import cv2  # OPENCV2
import cv2 as cv

import gym
import tensorflow as tf
import numpy as np
import random

from collections import deque



ENV_NAME = 'Breakout-v4'
EPISODE = 10000
CNN_INPUT_WIDTH = 80
CNN_INPUT_HEIGHT = 80

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.0001
OBSERVE_TIME = 500

STEP  = 1500
REPLAY_SIZE = 50000
BATCH_SIZE = 32
GAMMA = 0.99

# what's this?
SERIES_LENGTH = 4



class ImageProcess():
    def ColorMat2Binary(self, state):
        height = state.shape[0]
        width = state.shape[1]
        nchannel = state.shape[2]

        sHeight = int(height * 0.5)
        sWidth = CNN_INPUT_WIDTH

        state_gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)

        _, state_binary = cv2.threshold(state_gray, 5, 255, cv2.THRESH_BINARY)

        state_binarySmall = cv2.resize(state_binary, (sWidth, sHeight), interpolation=cv2.INTER_AREA)

        cnn_inputImg = state_binarySmall[25:, :]
        cnn_inputImg = cnn_inputImg.reshape((CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT))

        return cnn_inputImg


class DQN():
    def __init__(self, env):
        self.imageProcess = ImageProcess()
        self.epsilon = INITIAL_EPSILON
        self.action_dim = env.action_space.n
        self.observe_time = 0

        self.replay_buffer = deque()

        self.time_step = 0

        self.session = tf.InteractiveSession()
        self.create_network()

        self.session.run(tf.initialize_all_variables())

    def get_action(self, state_shadow):
        if self.epsilon >= FINAL_EPSILON and self.observe_time > OBSERVE_TIME:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000

        action = np.zeros(self.action_dim)
        action_index = None
        if random.random() < self.epsilon:
            action_index = random.randint(0, self.action_dim - 1)
        else:
            action_index = self.get_greedy_action(state_shadow)

        return action_index

    def get_greedy_action(self, state_shadow):
        rst = self.Q_value.eval(feed_dict={self.input_layer: [state_shadow]})[0]
        #print(np.max(rst))
        return np.argmax(rst)

    def create_network(self):

        INPUT_DEPTH = SERIES_LENGTH

        self.input_layer = tf.placeholder(tf.float32, [None, CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT, INPUT_DEPTH],
                                          name='status-input')
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])
        self.y_input = tf.placeholder(tf.float32, [None])

        W1 = self.get_weights([8, 8, 4, 32])
        b1 = self.get_bias([32])

        h_conv1 = tf.nn.relu(tf.nn.conv2d(self.input_layer, W1, strides=[1, 4, 4, 1], padding='SAME') + b1)
        conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W2 = self.get_weights([4, 4, 32, 64])
        b2 = self.get_bias([64])

        h_conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides=[1, 2, 2, 1], padding='SAME') + b2)

        W3 = self.get_weights([3, 3, 64, 64])
        b3 = self.get_bias([64])

        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3)

        W_fc1 = self.get_weights([1600, 512])
        b_fc1 = self.get_bias([512])

        conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(conv3_flat, W_fc1) + b_fc1)

        W_fc2 = self.get_weights([512, self.action_dim])
        b_fc2 = self.get_bias([self.action_dim])

        self.Q_value = tf.matmul(h_fc1, W_fc2) + b_fc2
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))

        self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    def percieve(self, state_shadow, action_index, reward, state_shadow_next, done, episode):

        action = np.zeros( self.action_dim )
        action[ action_index ] = 1

        self.replay_buffer.append([state_shadow, action, reward, state_shadow_next, done])

        self.observe_time += 1
        if self.observe_time % 1000 and self.observe_time <= OBSERVE_TIME == 0:
            #print(self.observe_time)
            pass

        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE and self.observe_time > OBSERVE_TIME:
            self.train_network()

    def train_network(self):
        self.time_step += 1

        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done_batch = [data[4] for data in minibatch]

        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.input_layer: next_state_batch})

        for i in range(BATCH_SIZE):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.input_layer: state_batch,
            self.action_input: action_batch,
            self.y_input: y_batch
        })

    def get_weights(self, shape):
        weight = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(weight)

    def get_bias(self, shape):
        bias = tf.constant(0.01, shape=shape)
        return tf.Variable(bias)



def main():
    env = gym.make(ENV_NAME)
    state_shadow = None
    next_state_shadow = None

    agent = DQN(env)
    total_reward_decade = 0

    for episode in range(EPISODE):

        total_reward = 0
        state = env.reset()
        state = agent.imageProcess.ColorMat2Binary(state)  # now state is a binary image of 80 * 80
        state_shadow = np.stack((state, state, state, state), axis=2)

        for step in range(STEP):
            env.render()
            action = agent.get_action(state_shadow)

            next_state, reward, done, _ = env.step(action)

            next_state = np.reshape(agent.imageProcess.ColorMat2Binary(next_state), (80,80,1))

            next_state_shadow = np.append(next_state, state_shadow[:,:,:3], axis=2)

            total_reward += reward
            agent.percieve(state_shadow, action, reward, next_state_shadow, done, episode)
            state_shadow = next_state_shadow

            if done:
                break

        print('Episode:', episode, 'Total Point in this Episode is:', total_reward)
        total_reward_decade += total_reward

        if episode % 10 == 0:
            print('-------------')
            print('Decade:', episode / 10, 'Total Reward in this Decade is:', total_reward_decade)
            print('-------------')
            total_reward_decade = 0



if __name__ == '__main__':
    main()
