import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from Boxer import Boxer
import constant


class DeepQNetwork:
    def __init__(
                self,
                n_agents,
                n_actions,
                n_features,
                load_model,
                path,
                output_graph,
                random_seed,
                n_agents_trained_model,
                random_seed_trained_model,
                number_episode
                ):
        self.n_agents = n_agents
        self.step = 0
        self.sess= tf.Session()

        self.Boxer_name = []
        for i in range(self.n_agents):
            self.Boxer_name.append("Boxer" + str(i+1))

        self.Boxer = []
        self.pre_train_steps = 2000
        for i in range(self.n_agents):
            self.Boxer.append(Boxer(self.sess,
                                    self.Boxer_name[i],
                                    n_actions,
                                    n_features,
                                    load_model,
                                    path,
                                    output_graph=output_graph,
                                    number_agent=n_agents,
                                    random_seed=random_seed,
                                    n_agents_trained_model=n_agents_trained_model,
                                    random_seed_trained_model=random_seed_trained_model,
                                    number_episode=number_episode
                                    ))

        writer = tf.summary.FileWriter("./NUM_AGENT" +
                                       str(n_agents).zfill(2) +
                                       "/" + str(random_seed).zfill(4) +
                                       '/logs', self.sess.graph)
        #writer = tf.summary.FileWriter("~/simulation/logs/", self.sess.graph)
        #self.saver = tf.train.Saver()
        #self.loading_parameters()


    def loading_parameters(self):
        self.saver.restore(self.sess, './IDETC_TEST' + '/job4' + '/model-' + str(5000) + 'Boxer1' + '.ckpt')
        self.saver.restore(self.sess, './IDETC_TEST' + '/job4' + '/model-' + str(5000) + 'Boxer2' + '.ckpt')
        self.saver.restore(self.sess, './IDETC_TEST' + '/job4' + '/model-' + str(5000) + 'Boxer3' + '.ckpt')
        self.saver.restore(self.sess, './IDETC_TEST' + '/job4' + '/model-' + str(5000) + 'Boxer4' + '.ckpt')
        # save_dict = {}
        # name = 'Boxer1'
        # for v in tf.all_variables():
        #     # print(v)
        #     if name in v.name:
        #         ckpt_var_name = v.name.strip(':0')
        #         # print(ckpt_var_name)
        #         ckpt_var_name = ckpt_var_name.replace(name, 'Boxer1')
        #         # print (name, ckpt_var_name)
        #         save_dict[ckpt_var_name] = v
        #     # else:
        #     #     print('else', v.name)
        #     #     save_dict[v.name.strip(':0')] = v
        # # print(tf.all_variables())
        # print(save_dict.keys())
        # saver1 = tf.train.Saver(var_list=save_dict)
        # saver1.restore(self.sess, './IDETC_TEST' + '/job3' + '/model-' + str(5000) + 'Boxer1' + '.ckpt')
        #
        # name = 'Boxer2'
        # # save_dict = {}
        # for v in tf.all_variables():
        #     # print(v)
        #     if name in v.name:
        #         ckpt_var_name = v.name.strip(':0')
        #         # print(ckpt_var_name)
        #         ckpt_var_name = ckpt_var_name.replace(name, 'Boxer2')
        #         # print (name, ckpt_var_name)
        #         save_dict[ckpt_var_name] = v
        #     # else:
        #     #     print('else', v.name)
        #     #     save_dict[v.name.strip(':0')] = v
        # # print(tf.all_variables())
        # print(save_dict.keys())
        # saver1 = tf.train.Saver(var_list=save_dict)
        # saver1.restore(self.sess, './IDETC_TEST' + '/job3' + '/model-' + str(5000) + 'Boxer2' + '.ckpt')
        #
        # name = 'Boxer3'
        # # save_dict = {}
        # for v in tf.all_variables():
        #     # print(v)
        #     if name in v.name:
        #         ckpt_var_name = v.name.strip(':0')
        #         # print(ckpt_var_name)
        #         ckpt_var_name = ckpt_var_name.replace(name, 'Boxer3')
        #         # print (name, ckpt_var_name)
        #         save_dict[ckpt_var_name] = v
        #     # else:
        #     #     print('else', v.name)
        #     #     save_dict[v.name.strip(':0')] = v
        # # print(tf.all_variables())
        # # print(save_dict.keys())
        # saver1 = tf.train.Saver(var_list=save_dict)
        # saver1.restore(self.sess, './IDETC_TEST' + '/job3' + '/model-' + str(5000) + 'Boxer3' + '.ckpt')
        #
        # name = 'Boxer4'
        # # save_dict = {}
        # for v in tf.all_variables():
        #     # print(v)
        #     if name in v.name:
        #         ckpt_var_name = v.name.strip(':0')
        #         # print(ckpt_var_name)
        #         ckpt_var_name = ckpt_var_name.replace(name, 'Boxer4')
        #         print(name, ckpt_var_name)
        #         save_dict[ckpt_var_name] = v
        #     # else:
        #     #     print('else', v.name)
        #     #     save_dict[v.name.strip(':0')] = v
        # # print(tf.all_variables())
        # # print(save_dict.keys())
        # print([v.name for v in tf.all_variables()])
        # saver1 = tf.train.Saver(var_list=save_dict)
        # saver1.restore(self.sess, './IDETC_TEST' + '/job4' + '/model-' + str(5000) + 'Boxer4' + '.ckpt')
        #
        # #for v in tf.all_variables():
        #     #print('V:')
        #     #print(v)
    def choose_action(self,observation):
        actions = []
        for i in range(self.n_agents):
            actions.append(self.Boxer[i].choose_action(observation[self.Boxer_name[i]]))

        action_list={}
        for i in range(self.n_agents):
            action_list.update({self.Boxer_name[i]:actions[i]})
            #print(action_list)
        return action_list

    def store_transition(self, s, a, r, s_, done):
        state = []
        action = []
        reward = []
        next_state = []
        done_list = []

        for i in range(self.n_agents):
            state.append(s[self.Boxer_name[i]])
            action.append(a[self.Boxer_name[i]])
            reward.append(r[self.Boxer_name[i]])
            next_state.append(s_[self.Boxer_name[i]])
            done_list.append(done[self.Boxer_name[i]])

        for i in range(self.n_agents):
            self.Boxer[i].store_transition(state[i],action[i], reward[i], next_state[i],done_list[i])

    def choose_action_greedy(self, observation, epsilon_greedy):
        actions = []
        for i in range(self.n_agents):
            actions.append(self.Boxer[i].choose_action_greedy(observation[self.Boxer_name[i]],epsilon_greedy))

        action_list = {}
        for i in range(self.n_agents):
            action_list.update({self.Boxer_name[i]: actions[i]})

        return action_list