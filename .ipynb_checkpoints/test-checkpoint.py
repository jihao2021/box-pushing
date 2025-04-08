from scripts.pygame_env import CustomEnv
from scripts.DQN_modified_plus_simplified import DeepQNetwork

def main():
    N_AGENTS = 2
    env = CustomEnv(random_initialization=False, n_agents=N_AGENTS)

    RL = DeepQNetwork(n_agents=N_AGENTS,
                      n_actions=env.action_space.n,
                      n_features=env.observation_space.shape[0],
                      load_model=False,
                      # path="./dqn",
                      path="./NUM_AGENT" + str(0).zfill(2) + "/" +
                           str(0).zfill(4) + '/saved_model',
                      output_graph=True,
                      random_seed=0,
                      n_agents_trained_model=0,
                      random_seed_trained_model=None,
                      number_episode=0
                      )

    env.render()


if __name__ == "__main__":
    main()