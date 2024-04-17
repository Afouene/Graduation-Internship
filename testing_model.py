from stable_baselines3 import PPO
from setting_the_environment import AUVEnvironment  
import numpy as np
import matplotlib.pyplot as plt

model_path="logs/26/rl_model_195000_steps.zip"

model = PPO.load(model_path)


env = AUVEnvironment()

num_episodes = 1000
average_age_over_episodes= []
average_energy_harvested_over_episodes=[]
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs,deterministic=False)  
        obs, reward, done, _ = env.step(action)  
        total_reward += reward

        #env.render()
        
    #print("the AOI is",env.AoI_all_nodes)
    average_age_over_episodes.append(np.mean(env.reward_per_step))
    average_energy_harvested_over_episodes.append(env.energy_harvested)
    #print("occurence",env.occurence)

    #print("This is the power transfered to nodes",env.cumulative_rewards)
    #print("This the average reward",abs(np.mean((env.reward_per_step))))
print("This is for the average age for RL algorithm ",env.num_devices," nodes",np.mean(average_age_over_episodes))
print("This is the average  cummulative energy harvested for RL algorithm ",env.num_devices," nodes",np.mean(average_energy_harvested_over_episodes))