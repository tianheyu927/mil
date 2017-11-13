import gym
import numpy as np
import tensorflow as tf
import pickle
import os
import time
import imageio
from utils import mkdir_p, load_scale_and_bias

REACH_SUCCESS_THRESH = 0.05
REACH_SUCCESS_TIME_RANGE = 10
REACH_DEMO_CONDITIONS = 10

def evaluate_vision_reach(env, graph, model, data_generator, sess, exp_string, record_gifs, log_dir):
    T = model.T
    scale, bias = load_scale_and_bias('data/scale_and_bias_sim_vision_reach.pkl')
    successes = []
    selected_demo = data_generator.selected_demo
    if record_gifs:
        record_gifs_dir = os.path.join(log_dir, 'evaluated_gifs')
        mkdir_p(record_gifs_dir)
    for i in xrange(len(selected_demo['selected_demoX'])):
        selected_demoO = selected_demo['selected_demoO'][i]
        selected_demoX = selected_demo['selected_demoX'][i]
        selected_demoU = selected_demo['selected_demoU'][i]
        if record_gifs:
            gifs_dir = os.path.join(record_gifs_dir, 'color_%d' % i)
            mkdir_p(gifs_dir)
        for j in xrange(REACH_DEMO_CONDITIONS):
            if j in data_generator.demos[i]['demoConditions']:
                dists = []
                # ob = env.reset()
                # use env.set_state here to arrange blocks
                Os = []
                for t in range(T):
                    # import pdb; pdb.set_trace()
                    env.render()
                    time.sleep(0.05)
                    obs, state = env.env.get_current_image_obs()
                    Os.append(obs)
                    obs = np.transpose(obs, [2, 1, 0]) / 255.0
                    obs = obs.reshape(1, 1, -1)
                    state = state.reshape(1, 1, -1)
                    feed_dict = {
                        model.obsa: selected_demoO,
                        model.statea: selected_demoX.dot(scale) + bias,
                        model.actiona: selected_demoU,
                        model.obsb: obs,
                        model.stateb: state.dot(scale) + bias
                    }
                    with graph.as_default():
                        action = sess.run(model.test_act_op, feed_dict=feed_dict)
                    ob, reward, done, reward_dict = env.step(np.squeeze(action))
                    dist = -reward_dict['reward_dist']
                    if t >= T - REACH_SUCCESS_TIME_RANGE:
                        dists.append(dist)
                if np.amin(dists) <= REACH_SUCCESS_THRESH:
                    successes.append(1.)
                else:
                    successes.append(0.)
                if record_gifs:
                    video = np.array(Os)
                    record_gif_path = os.path.join(gifs_dir, 'cond%d.samp0.gif' % j)
                    print 'Saving gif sample to :%s' % record_gif_path
                    imageio.mimwrite(record_gif_path, video)
            env.render(close=True)
            if j != REACH_DEMO_CONDITIONS - 1 or i != len(selected_demo['selected_demoX']) - 1:
                env.env.next()
                env.render()
                time.sleep(0.5)
        if i % 5  == 0:
            print "Task %d: current success rate is %.5f" % (i, np.mean(successes))
    success_rate_msg = "Final success rate is %.5f" % (np.mean(successes))
    print success_rate_msg
    with open('logs/log_sim_vision_reach.txt', 'a') as f:
        f.write(exp_string + ':\n')
        f.write(success_rate_msg + '\n')
        