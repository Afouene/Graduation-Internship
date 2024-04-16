from stable_baselines3 import PPO
from setting_the_environment import AUVEnvironment  
import numpy as np
import matplotlib.pyplot as plt

model_path="logs/11/rl_model_255000_steps.zip"
model = PPO.load(model_path)


env = AUVEnvironment()

num_episodes = 100
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
        

    """
        print("action",action)
        #print("dis",np.linalg.norm(env.sensor_node_positions[action[1]] - env.auv_position))
    print("the aoi of each node",env.AoI_all_nodes)
    print("the average aoi is ",np.sum(env.AoI_all_nodes)/5)
    print(f"Episode {episode + 1}: Total reward = {total_reward}")
    print('total rewards for all nodes',env.cumulative_rewards)"""
    #print("the AOI is",env.AoI_all_nodes)
    average_age_over_episodes.append(np.mean(env.reward_per_step))
    average_energy_harvested_over_episodes.append(env.energy_harvested)

    #print("This is the power transfered to nodes",env.cumulative_rewards)
    #print("This the average reward",abs(np.mean((env.reward_per_step))))
print("This is for the average age",np.mean(average_age_over_episodes))
print("This is the average  cummulative energy harvested",np.mean(average_energy_harvested_over_episodes))