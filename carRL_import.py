import tensorflow as tf
import numpy as np

def run_model(state):
    dir = './model20190427-18-51/trainedModel4589/'     # directory of the trained model

    sess = tf.Session()
    saver = tf.train.import_meta_graph(dir + 'RLmodel_trianed.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(dir))

    #ls = [n.name for n in tf.get_default_graph().as_graph_def().node]
    #print(ls)
    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name('Placeholder:0')    # name found by "print(model._states)" in carRL.py
    y = graph.get_tensor_by_name('dense_3/BiasAdd:0')    # name found by "print(model._logits)" in carRL.py

    #state : [position, position_y, velocity, theta, goal_position, goal_position_y]

    output = sess.run(y, feed_dict={x:state})   # calculate predicted reward for each action
    action_chosen = np.argmax(output)   # choose action with biggest reward calculated
    return action_chosen

i = [1,1,0,0,11,-11]   # current state
b = np.array(i).reshape((1,6))

action = run_model(b)
print(action)
# 0: left turn, decelerate
# 1: left turn, no change in speed
# 2: left turn, accelerate
# 3: no turn, decelerate
# 4: no turn, no change in speed
# 5: no turn, accelerate
# 6: right turn, decelerate
# 7: right turn, no change in speed
# 8: right turn, accelerate
