import tensorflow as tf
import numpy as np
import carRL_env as environment
import math

def run(env):
    state = env.reset()
    #state : [position, position_y, velocity, theta, goal_position, goal_position_y]

    tot_reward = 0
    max_x = -100
    render = True
    steps = 0
    done = False

    dir = './model20190427-18-51/trainedModel1015/'     # directory of the trained model
    sess = tf.Session()
    saver = tf.train.import_meta_graph(dir + 'RLmodel_trianed.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(dir))

    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name('Placeholder:0')    # name found by "print(model._states)" in carRL.py
    y = graph.get_tensor_by_name('dense_3/BiasAdd:0')    # name found by "print(model._logits)" in carRL.py

    #state : [position, position_y, velocity, theta, goal_position, goal_position_y]
    num_episodes = 1
    cnt = 0
    while cnt < num_episodes:
        while True:
            if render:
                env.render()
            #state = state.reshape((1,6))
            output = sess.run(y, feed_dict={x:state.reshape((1,6))})   # calculate predicted reward for each action
            action = np.argmax(output)   # choose action with biggest reward calculated

            next_state, reward, done, info = env.step(action)
            next_state = np.array(next_state).reshape((1,6))
            if done:
                next_state = None

            # exponentially decay the eps value
            steps += 1

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            if done:
                cnt += 1
                state = env.reset()
                state = np.array(state).reshape((1,6))
                break

env_a = environment.MountainCarEnv()
run(env_a)
