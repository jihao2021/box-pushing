##Boxer is an abstraction of neural network class for box-pushing agent

#import pygame as pg
import tensorflow as tf
import numpy as np
import os
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt

#from tensorflow.contrib.framework.python.framework import checkpoint_utils
#import constant

#np.random.seed(constant.random_seed)
#tf.set_random_seed(constant.random_seed)

class MyModel(keras.Model):

    def __init__(self, size=16, num_actions=6):
        super().__init__()
        self.dense1 = layers.Dense(size, activation='relu')
        # self.dense2 = layers.Dense(size, activation='relu')

        # Value stream
        self.value_fc = layers.Dense(size, activation='relu')
        self.value = layers.Dense(1)

        # Advantage stream
        self.advantage_fc = layers.Dense(size, activation='relu')
        self.advantage = layers.Dense(num_actions)

    def call(self, inputs):
        # Shared layers
        x = self.dense1(inputs)
        # x = self.dense2(x)

        # Value stream
        # value = self.value_fc(x)
        value = self.value(x)

        # Advantage stream
        # advantage = self.advantage_fc(x)
        advantage = self.advantage(x)

        # Combine value and advantage into Q-values
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        return q_values


class Boxer():
    
    share_num = 0 #class variable shared by all instance

    def __init__(self,
                 name, 
                 n_actions, 
                 n_features, 
                 load_model,
                 path,
                 output_graph, 
                 number_agent,
                 random_seed,
                 n_agents_trained_model,
                 random_seed_trained_model,
                 number_episode,
                 distribution
                 ):
        self.name = name
        self.step = 0
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = 0.001
        self.gamma = 0.99
        self.startE = 1.0
        self.replace_target_iter = 200
        self.memory_size = 1000  #previous is 2000
        self.batch_size = 32
        self.load_model = load_model
        self.path = path
        self.anneling_steps = 100000
        self.epsilon = self.startE
        self.endE = 0.005   # previous is 0.01
        self.stepDrop = (self.startE - self.endE) / self.anneling_steps
        self.pre_train_steps = 2000
        self.dueing = True
        self.dueing_plus_double = False

        self.number_agent = number_agent
        self.random_seed = random_seed
        self.n_agents_trained_model=n_agents_trained_model
        self.random_seed_trained_model=random_seed_trained_model
        self.number_episode=number_episode
        self.distribution=self.generate_values(distribution)

        '''total learning step'''
        self.learn_step_counter = 0

        '''initialize zero memory[s,a,r,s_], action is 1(select one action), reward is 1, old and new states'''
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 3))

        self._build_net_new(name)

        # t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        # e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        # self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # if not os.path.exists(self.path):
        #     pass
        #     os.makedirs(self.path)

        # self.saver = tf.train.Saver()
        if self.load_model == True:
            pass
        else:
            print('Error in initializing parameters in Boxer')
            
        self.cost_his = []

        if self.load_model:
            #self.loading3_4_new(name, self.n_agents_trained_model, self.random_seed_trained_model)
            #print('yes yes yes')
            
            if self.number_agent < self.n_agents_trained_model:
                self.loading_others_higher(others_number=self.n_agents_trained_model,
                                           others_random_seed=self.random_seed_trained_model)
            elif self.number_agent > self.n_agents_trained_model:
                self.loading_others_lower(others_number=self.n_agents_trained_model, 
                                      others_random_seed=self.random_seed_trained_model)
            elif self.number_agent == self.n_agents_trained_model:
                self.loading_self(others_number=self.n_agents_trained_model,
                                others_random_seed=self.random_seed_trained_model)
            else:
                print('Error in loading model in boxer.py file')
            


            '''
            if self.number_agent != self.n_agents_trained_model:
                self.loading_semifinal(others_number=self.n_agents_trained_model,
                                           others_random_seed=self.random_seed_trained_model)
            elif self.number_agent == self.n_agents_trained_model:
                self.loading_self()
            else:
                print('Error in laoding model in boxer.py file')
            '''

        self.one_hit = 0  ##right-top corner hit
        self.two_hit = 0  ##right-down corner hit
        self.three_hit = 0  ##left-down corner hit
        self.four_hit = 0  ##left-top corner hit
        self.five_hit = 0  ## top hit
        self.six_hit = 0  ##bottom hit

    def get_share_num(self):
        return Boxer.share_num

    def set_share_num(value):
        Boxer.share_num = value

    def _build_net_new(self, name, reuse=False):
        n_l1, n_l2, w_initializer, b_initializer = 16, 32, tf.random_normal_initializer(0., 0.3), \
                                                   tf.constant_initializer(0.1)

        self.eval_net = MyModel(size=n_l1, num_actions=self.n_actions)
        self.target_net = MyModel(size=n_l1, num_actions=self.n_actions)

        input_shape = (None, self.n_features)
        self.eval_net.build(input_shape=input_shape)
        self.target_net.build(input_shape=input_shape)

        # Print model summaries
        # self.eval_net.summary()
        # self.target_net.summary()

        # self.loss = tf.math.reduce_mean(tf.math.squared_difference(self.eval_net(observation), self.target_net(observation)))
        self.optimizer = tf.keras.optimizers.Adam(self.lr)

#     def _build_net(self, name, reuse=False):
#         # if reuse:

#         self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s'+name)
#         self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target'+name)
#         n_l1, n_l2, w_initializer, b_initializer = 16, 32, tf.random_normal_initializer(0., 0.3), \
#                                                    tf.constant_initializer(0.1)

#         with tf.variable_scope('eval_net'+name):

#             l1 = tf.layers.dense(self.s, n_l1, tf.nn.relu, kernel_initializer=w_initializer,
#                                  bias_initializer=b_initializer, name='l1')

#             with tf.variable_scope('Value'):
#                 w2 = tf.get_variable('w2', [n_l1,1], initializer=w_initializer)
#                 b2 = tf.get_variable('b2', [1,1], initializer=b_initializer)
#                 self.V = tf.matmul(l1,w2) + b2

#             with tf.variable_scope('Advantage'):
#                 w2 = tf.get_variable('w2',[n_l1, self.n_actions],initializer=w_initializer)
#                 b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer)
#                 self.A = tf.matmul(l1,w2)+b2

#             with tf.variable_scope('Q'):
#                 self.q_eval = self.V + (self.A - tf.reduce_mean(self.A, axis=1,keep_dims=True))

#         with tf.variable_scope('loss'+name):
#             self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
#         with tf.variable_scope('train'+name):
#             self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

#         """Build target_net"""

#         self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_'+name)

#         with tf.variable_scope('target_net'+name):
#             l1 = tf.layers.dense(self.s_, n_l1, tf.nn.relu, kernel_initializer=w_initializer,
#                                  bias_initializer=b_initializer, name='l1')

#             with tf.variable_scope('Value'):
#                 w2 = tf.get_variable('w2', [n_l1,1], initializer=w_initializer)
#                 b2 = tf.get_variable('b2', [1,1], initializer=b_initializer)
#                 self.V = tf.matmul(l1,w2) + b2

#             with tf.variable_scope("Advantage"):
#                 w2 = tf.get_variable('w2',[n_l1, self.n_actions],initializer=w_initializer)
#                 b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer)
#                 self.A = tf.matmul(l1,w2)+b2

#             with tf.variable_scope('Q'):
#                 self.q_next = self.V + (self.A - tf.reduce_mean(self.A, axis=1,keep_dims=True))

    def store_transition(self, s, a, r, s_, done):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a,r], s_, done))

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if self.step >= self.pre_train_steps:
            if self.epsilon > self.endE:
                self.epsilon -= self.stepDrop
        if np.random.uniform() < self.epsilon or self.step < self.pre_train_steps:
            # Choose the row based on prob_num
            #row_index = self.get_share_num() % self.distribution.shape[0]
            #chosen_row = self.distribution[row_index]
            #action = np.random.choice(range(len(chosen_row)), p=chosen_row)

            action = np.random.randint(0, self.n_actions)
        else:
            actions_value = self.eval_net(observation)
            action = np.argmax(actions_value)

        return action

    def get_qmax(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.eval_net(observation)
        qmax = np.max(actions_value)
        return qmax
    
    def generate_values_helper(self):
        value1 = np.random.uniform(0, 0.12)
        remaining_sum = 1 - value1
        values = np.random.rand(3)
        values /= values.sum()  # Normalize to sum to 1
        values *= remaining_sum  # Scale to the remaining_sum
        all_values = np.concatenate(([value1], values))
        while abs(all_values[0] - all_values[2]) < 0.15:
            value1 = np.random.uniform(0, 0.12)
            remaining_sum = 1 - value1
            values = np.random.rand(3)
            values /= values.sum()
            values *= remaining_sum
            all_values = np.concatenate(([value1], values))
        result_4x4 = np.zeros((4, 4))
        for i in range(4):
            result_4x4[i] = np.roll(all_values, shift=i)
        result_4x6 = np.zeros((4, 6))
        result_4x6[:, 0] = result_4x4[:, 0]  # 0th column of 4x6 is the same as 0th column of 4x4
        result_4x6[:, 1] = result_4x4[:, 1] / 2  # 1st column of 4x6 is half of 1st column of 4x4
        result_4x6[:, 2] = result_4x4[:, 1] / 2 # 2nd column of 4x6 is the same as 2nd column of 4x4
        result_4x6[:, 3] = result_4x4[:, 2]  # 3rd column of 4x6 is half of 3rd column of 4x4
        result_4x6[:, 4] = result_4x4[:, 3] / 2  # 4th column of 4x6 is half of 4th column of 4x4
        result_4x6[:, 5] = result_4x4[:, 3] / 2 # 5th column of 4x6 is the same as 4th column of 4x4

        row_sums = result_4x6.sum(axis=1)
        result_4x6 /= row_sums[:, np.newaxis]  # Divide each row by its sum

        return result_4x6
    
    def generate_values(self, distribution):
        combined_p_arrays = np.zeros((0, 6))

        for i in range(distribution):
            p_arrays_temp = self.generate_values_helper()
            combined_p_arrays = np.vstack((combined_p_arrays, p_arrays_temp))

        return combined_p_arrays

    def choose_action_greedy(self, observation, epsilon_greedy):
        observation = observation[np.newaxis, :]
        self.epsilon = epsilon_greedy

        if np.random.uniform() < self.epsilon:
            # Choose the row based on prob_num
            #row_index = self.get_share_num() % self.distribution.shape[0]
            #chosen_row = self.distribution[row_index]
            #action = np.random.choice(range(len(chosen_row)), p=chosen_row)

            action = np.random.randint(0, self.n_actions)
            
        else:
            actions_value = self.eval_net(observation)
            action = np.argmax(actions_value)
        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.set_weights(self.eval_net.get_weights())

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]


        # train model
        with tf.GradientTape() as tape:
            s_batch = batch_memory[:,:self.n_features]
            s_next_batch = batch_memory[:, -(self.n_features+1):-1]
            if self.dueing_plus_double:
                q_next = self.target_net(s_next_batch)
                q_eval4next = self.eval_net(s_next_batch)
                q_eval = self.eval_net(s_batch)
            else:
                q_next = self.target_net(s_next_batch)
                q_eval = self.eval_net(s_batch)

            # print(q_eval)
            q_target = tf.identity(q_eval)
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, self.n_features].astype(int)
            reward = batch_memory[:, self.n_features + 1]
            done = batch_memory[:, -1]

            # if self.dueing_plus_double:
            #     max_act4next = np.argmax(q_eval4next, axis=1)
            #     selected_q_next = q_next[batch_index, max_act4next]
            # else:
            #     selected_q_next = np.max(q_next, axis=1)

            if self.dueing_plus_double:
                max_act4next = np.argmax(q_eval4next, axis=1)
                selected_q_next = q_next.numpy()[batch_index, max_act4next]
            else:
                selected_q_next = np.max(q_next.numpy(), axis=1)


            q_target = q_target.numpy()
            q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next * (1 - done)
            q_target = tf.convert_to_tensor(q_target)

            loss = tf.reduce_mean(tf.square(q_eval - q_target))
            gradients = tape.gradient(loss, self.eval_net.trainable_variables)

            # Debug gradients
            for var, grad in zip(self.eval_net.trainable_variables, gradients):
                if grad is None:
                    print(f"Warning: No gradient for variable {var.name}")

            if any(grad is None for grad in gradients):
                raise ValueError("Gradient is None for one or more variables")

            self.optimizer.apply_gradients(
                zip(gradients, self.eval_net.trainable_variables))

            self.cost_his.append(loss)

        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his), self.cost_his))
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

    # def loading_self(self, others_number, others_random_seed):
    #     #print('Loading Model...')
    #     #self.saver.restore(self.sess, self.path + '/model-' + str(self.number_episode) + self.name + '.ckpt')
    #     name = self.name
    #     print('Loading Model...')
    #     load_path = "./NUM_AGENT" +str(others_number).zfill(2) +"/" + str(others_random_seed).zfill(4) +'/saved_model'
    #     self.saver.restore(self.sess, load_path + '/model-' + str(self.number_episode) + self.name + '.ckpt')

    # def loading_others_higher(self, others_number, others_random_seed):
    #     name = self.name
    #     print('Loading Model...')
    #     load_path = "./NUM_AGENT" +str(others_number).zfill(2) +"/" + str(others_random_seed).zfill(4) +'/saved_model'
    #     self.saver.restore(self.sess, load_path + '/model-' + str(self.number_episode) + self.name + '.ckpt')

    # def loading_others_lower(self, others_number, others_random_seed):
    #     print('Loading Model...')
    #     name = self.name
    #     load_path = "./NUM_AGENT" +str(others_number).zfill(2) +"/" + str(others_random_seed).zfill(4) +'/saved_model'
    #     Boxer_list_mine = []
    #     for i in range(self.number_agent):
    #         Boxer_list_mine.append('Boxer' + str(i+1))
    #
    #     Boxer_list_lower = []
    #     for i in range(others_number):
    #         Boxer_list_lower.append('Boxer' + str(i+1))
    #
    #     Boxer_list_lower = Boxer_list_lower * (int(self.number_agent / others_number) + 1)
    #     print('resized list before slicing is', len(Boxer_list_lower))
    #     Boxer_list_lower = Boxer_list_lower[:self.number_agent]
    #     print('final list after slicing is', len(Boxer_list_lower))
    #
    #     ###if i make ### delete, that is most recent delete
    #
    #
    #     for i in range(self.number_agent):
    #         if name in Boxer_list_mine[i:]:
    #             save_dict = {}
    #             for v in tf.all_variables():
    #                 #print(v.name)
    #                 if name in v.name:
    #                     #print(v.name)
    #
    #                     ckpt_var_name = v.name.strip(':0')
    #                     #print(ckpt_var_name)
    #                     ckpt_var_name = ckpt_var_name.replace(name, Boxer_list_lower[i])
    #                     # print (name, ckpt_var_name)
    #                     save_dict[ckpt_var_name] = v
    #                 #else:
    #                     #print('else', v.name)
    #                     #save_dict[v.name.strip(':0')] = v
    #             # print(tf.all_variables())
    #             #print(save_dict)
    #             ###print(save_dict.keys())
    #
    #             saver1 = tf.train.Saver(var_list=save_dict)
    #
    #             saver1.restore(self.sess, load_path + '/model-' + str(self.number_episode) + Boxer_list_lower[i] + '.ckpt')


if __name__ == "__main__":
    # model = MyModel()
    batch = np.ones([200, 9*2+3])
    # result = model(batch)
    # print(result)

    boxer = Boxer("test",
                  n_actions=6,
                  n_features=9,
                  load_model=False,
                  path=None,
                  output_graph=False,
                  number_agent=False,
                  random_seed=False,
                  n_agents_trained_model=False,
                  random_seed_trained_model=False,
                  number_episode=1
                  )

    print("before learning", boxer.eval_net.get_weights()[0])
    boxer.memory = batch
    boxer.memory_counter = 200
    boxer.learn()

    print("after learning", boxer.eval_net.get_weights()[0])