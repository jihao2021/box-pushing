import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from .Boxer import Boxer
from . import constant


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
        self.num_episodes = number_episode
        self.n_agents = n_agents
        self.step = 0
        #         self.sess= tf.Session()

        self.Boxer_name = []
        for i in range(self.n_agents):
            self.Boxer_name.append("Boxer" + str(i + 1))

        self.Boxer = []
        self.pre_train_steps = 2000
        for i in range(self.n_agents):
            self.Boxer.append(Boxer(self.Boxer_name[i],
                                    n_actions=n_actions,
                                    n_features=n_features,
                                    load_model=load_model,
                                    path=path,
                                    output_graph=output_graph,
                                    number_agent=n_agents,
                                    random_seed=random_seed,
                                    n_agents_trained_model=n_agents_trained_model,
                                    random_seed_trained_model=random_seed_trained_model,
                                    number_episode=number_episode
                                    )
                              )

        # writer = tf.summary.FileWriter("./NUM_AGENT" +
        #                                str(n_agents).zfill(2) +
        #                                "/" + str(random_seed).zfill(4) +
        #                                '/logs', self.sess.graph)

    def save_models(self):
        """
        Save weights to `path`
        """
        parent_dir = f"models/job{self.n_agents}/"

        # parent_dir = os.path.dirname(path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        # Save the weights
        for i in range(self.n_agents):
            self.Boxer[i].eval_net.save_weights(f"models/job{self.n_agents}/model-{self.num_episodes}Boxer{i}-eval.h5")
            self.Boxer[i].target_net.save_weights(
                f"models/job{self.n_agents}/model-{self.num_episodes}Boxer{i}-target.h5")

    def loading_parameters(self, num_trained, random_seed):
        if num_trained == self.n_agents:  # same number of agents
            print(f"loading exact {num_trained} trained models")
            for i in range(self.n_agents):
                self.Boxer[i].eval_net.load_weights(f"models/job{self.n_agents}/model-{self.num_episodes}Boxer{i}-eval.h5")
                self.Boxer[i].target_net.load_weights(
                    f"models/job{self.n_agents}/model-{self.num_episodes}Boxer{i}-target.h5")

        elif num_trained < self.n_agents:
            np.random.seed(random_seed)
            idx = list(range(num_trained))
            sampled_idx = list(np.random.choice(idx, self.n_agents-num_trained))
            idx = idx + sampled_idx

            print(f"sampling from {num_trained} trained models to fit team of {self.n_agents}")
            for i, id in enumerate(idx):
                self.Boxer[i].eval_net.load_weights(
                    f"models/job{num_trained}/model-{self.num_episodes}Boxer{id}-eval.h5")
                self.Boxer[i].target_net.load_weights(
                    f"models/job{num_trained}/model-{self.num_episodes}Boxer{id}-target.h5")

        else:  # if num_trained > self.n_agents
            np.random.seed(random_seed)
            idx = list(range(num_trained))
            sampled_idx = list(np.random.choice(idx, self.n_agents))
            idx = sampled_idx

            print(f"sampling from {num_trained} trained models to fit team of {self.n_agents}")
            for i, id in enumerate(idx):
                self.Boxer[i].eval_net.load_weights(
                    f"models/job{num_trained}/model-{self.num_episodes}Boxer{id}-eval.h5")
                self.Boxer[i].target_net.load_weights(
                    f"models/job{num_trained}/model-{self.num_episodes}Boxer{id}-target.h5")

    def choose_action(self, observation):
        actions = []
        for i in range(self.n_agents):
            actions.append(self.Boxer[i].choose_action(observation[self.Boxer_name[i]]))

        action_list = {}
        for i in range(self.n_agents):
            action_list.update({self.Boxer_name[i]: actions[i]})
            # print(action_list)
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
            self.Boxer[i].store_transition(state[i], action[i], reward[i], next_state[i], done_list[i])

    def choose_action_greedy(self, observation, epsilon_greedy):
        actions = []
        for i in range(self.n_agents):
            actions.append(self.Boxer[i].choose_action_greedy(observation[self.Boxer_name[i]], epsilon_greedy))

        action_list = {}
        for i in range(self.n_agents):
            action_list.update({self.Boxer_name[i]: actions[i]})

        return action_list
