from stable_baselines3 import PPO
from setting_the_environment import AUVEnvironment
import numpy as np
import matplotlib.pyplot as plt
from Round_Robin import RoundRobinAgent  # Importing Round Robin Agent class

# RL Algorithm
model_path = "logs/41/rl_model_955000_steps.zip"
model = PPO.load(model_path)
env_rl = AUVEnvironment()

# Random Selection Algorithm
env_random = AUVEnvironment()

# Round Robin Algorithm
env_round_robin = AUVEnvironment()
round_robin_agent = RoundRobinAgent(env_round_robin.num_devices)

num_episodes = 300
jain_index_rl = []
jain_index_random = []
jain_index_round_robin = []

for episode in range(num_episodes):
    # RL Algorithm
    obs_rl = env_rl.reset()
    done_rl = False
    while not done_rl:
        action_rl, _ = model.predict(obs_rl, deterministic=False)
        obs_rl, _, done_rl, _ = env_rl.step(action_rl)
    sum_of_squares_rl = sum(x ** 2 for x in env_rl.occurence)
    sum_of_values_rl = sum(env_rl.occurence)
    Jain_index_rl = ((sum_of_values_rl ** 2) / (sum_of_squares_rl * env_rl.num_devices)) if sum_of_squares_rl != 0 else 0
    jain_index_rl.append(Jain_index_rl)

    # Random Selection Algorithm
    done_random = False
    while not done_random:
        action_random = env_random.action_space.sample()
        _, _, done_random, _ = env_random.step(action_random)
    sum_of_squares_random = sum(x ** 2 for x in env_random.occurence)
    sum_of_values_random = sum(env_random.occurence)
    Jain_index_random = ((sum_of_values_random ** 2) / (sum_of_squares_random * env_random.num_devices)) if sum_of_squares_random != 0 else 0
    jain_index_random.append(Jain_index_random)

    # Round Robin Algorithm
    done_round_robin = False
    while not done_round_robin:
        action_round_robin = round_robin_agent.select_action()
        _, _, done_round_robin, _ = env_round_robin.step(action_round_robin)
    sum_of_squares_round_robin = sum(x ** 2 for x in env_round_robin.occurence)
    sum_of_values_round_robin = sum(env_round_robin.occurence)
    Jain_index_round_robin = ((sum_of_values_round_robin ** 2) / (sum_of_squares_round_robin * env_round_robin.num_devices)) if sum_of_squares_round_robin != 0 else 0
    jain_index_round_robin.append(Jain_index_round_robin)

# Plot Jain index over episodes for all algorithms
plt.plot(range(num_episodes), jain_index_rl, label='RL Algorithm')
plt.plot(range(num_episodes), jain_index_random, label='Random Selection Algorithm')
plt.plot(range(num_episodes), jain_index_round_robin, label='Round Robin Algorithm')
plt.xlabel('Episodes')
plt.ylabel('Jain Index')
plt.title('Jain Index  for 5 nodes : RL vs RW vs RR')
plt.legend()
plt.show()
