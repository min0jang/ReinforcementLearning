
class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self._num_states = num_states
        self._num_actions = num_actions
        self._batch_size = batch_size
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None
        # now setup the model
        self._define_model()

    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32, name='inputs')
        self._states = tf.placeholder(shape=[None, self._num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._num_actions], dtype=tf.float32)
        # create fully connected hidden layers
        fc1 = tf.layers.dense(self._states, 150, activation=tf.nn.relu, name='inputnode')
        fc2 = tf.layers.dense(fc1, 150, activation=tf.nn.relu)
        fc3 = tf.layers.dense(fc2, 100, activation=tf.nn.relu)
        self._logits = tf.layers.dense(fc3, self._num_actions, name='outputnode')
        self._logits = tf.layers.dense(fc3, self._num_actions)
        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states:
                                                 state.reshape(1, self._num_states)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})


class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)


class GameRunner:
    def __init__(self, sess, model, env, memory, max_eps, min_eps,
                 decay, render=True):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._render = render
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._eps = self._max_eps
        self._steps = 0
        self._reward_store = []
        self._max_x_store = []

    def run(self):
        state = self._env.reset()
        tot_reward = 0
        max_x = -100
        while True:
            if self._render:
                self._env.render()
            action = self._choose_action(state)

            next_state, reward, done, info = self._env.step(action)

            if done:
                next_state = None

            self._memory.add_sample((state, action, reward, next_state))
            self._replay()

            # exponentially decay the eps value
            self._steps += 1
            self._eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) \
                                      * math.exp(-LAMBDA * self._steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            if done:
                self._reward_store.append(tot_reward)
                self._max_x_store.append(max_x)
                break

        print("Step {}, Total penalty: {}, Epsilon: {}".format(self._steps, tot_reward, self._eps))

    def _choose_action(self, state):
        if random.random() < self._eps:
            return random.randint(0, self._model._num_actions - 1)
        else:
            return np.argmax(self._model.predict_one(state, self._sess))

    def _replay(self):
        batch = self._memory.sample(self._model._batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self._model._num_states)
                                 if val[3] is None else val[3]) for val in batch])

        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)

        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)

        # setup training arrays
        x = np.zeros((len(batch), self._model._num_states))
        y = np.zeros((len(batch), self._model._num_actions))

        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self._model.train_batch(self._sess, x, y)


import gym
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import random
import math
import carRL_env as environment
import datetime
import os

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0001
GAMMA = 0.99
BATCH_SIZE = 5

if __name__ == "__main__":
    env = environment.MountainCarEnv()

    num_states = 6
    num_actions = 9

    model = Model(num_states, num_actions, BATCH_SIZE)
    #print(model._states)    # print the name of the input node (for later use in imported version)
    #print(model._logits)    # print the name of the output node (for later use in imported version)
    mem = Memory(5)

    with tf.Session() as sess:
        sess.run(model._var_init)
        gr = GameRunner(sess, model, env, mem, MAX_EPSILON, MIN_EPSILON,
                        LAMBDA)
        num_episodes = 2000
        cnt = 0
        now = datetime.datetime.now()
        time = now.strftime("%Y%m%d-%H-%M")
        os.mkdir("./model{0}".format(time))
        while cnt < num_episodes:
            #if cnt % 5 == 0:
            print('  - Episode {} of {} -'.format(cnt+1, num_episodes))
            print('.')
            # if cnt == 100:
            #     gr._render = True

            gr.run()
            #print(gr._reward_store[cnt])
            qualification = -100

            if cnt > 10:
                if (gr._reward_store[cnt] > qualification) and\
                    (gr._reward_store[cnt-1] > qualification) and\
                    (gr._reward_store[cnt-2] > qualification) and\
                    (gr._reward_store[cnt-3] > qualification) and\
                    (gr._reward_store[cnt-4] > qualification) and\
                    (gr._reward_store[cnt-5] > qualification) and\
                    (gr._reward_store[cnt-6] > qualification) and\
                    (gr._reward_store[cnt-7] > qualification) and\
                    (gr._reward_store[cnt-8] > qualification) and\
                    (gr._reward_store[cnt-9] > qualification) and\
                    (gr._reward_store[cnt-10] > qualification) and\
                    (gr._reward_store[cnt-11] > qualification):
                    saver = tf.train.Saver()
                    episodeNumber = "{}".format(cnt)
                    saver.save(sess, "./model{0}/trainedModel{1}/RLmodel_trianed.ckpt".format(time,episodeNumber))
                    print("model {} saved".format(episodeNumber))

            cnt += 1



        plt.plot(gr._reward_store)
        plt.show()
