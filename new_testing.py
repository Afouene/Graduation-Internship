from stable_baselines3 import PPO
from new_env import AUVEnvironment  
import numpy as np
import matplotlib.pyplot as plt
import csv

model_path="logs/2/rl_model_13880000_steps.zip" #best one for 10 nodes till now
model_path="logs/3/rl_model_14640000_steps.zip" 
model_path="logs/3/rl_model_15000000_steps.zip" #this one is also good for 10
model_path="logs/4/rl_model_9360000_steps.zip" # 21 may 7 nodes the first 7nodes try 
model_path="logs/5/rl_model_12000000_steps.zip" # 21 may 7 nodes the first 7nodes try 
model_path="logs/6/rl_model_11120000_steps.zip" # 22 may 7 nodes eh not bad one 
model_path="logs/6/rl_model_8320000_steps.zip" # 
model_path="logs/7/rl_model_15000000_steps.zip" # 7nodes best one ppo 8
model_path="logs/8/rl_model_10000000_steps.zip" # 5nodes best one ppo 9
model_path="logs/9/rl_model_20000000_steps.zip" # 10nodes best one log 9ppo 10
model_path="logs/10/rl_model_2080000_steps.zip" # 3nodes best one ppo 11
model_path="logs/11/rl_model_13240000_steps.zip" # 7nodes best one ppo 12

model = PPO.load(model_path)


env = AUVEnvironment()
num_episodes = 1
average_age_over_episodes= []
average_energy_harvested_over_episodes=[]
jain_index_over_episodes = []  
auv_positions=[]
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs,deterministic=True)  
        obs, reward, done, _ = env.step(action)  
        total_reward += reward
        x=env.auv_position.copy()
        env.render()
        #print("auv position",env.auv_position)
        #print("action",action)
        auv_positions.append(x)

    sum_of_squares = sum(x**2 for x in env.occurence)
    sum_of_values = sum(env.occurence)

    Jain_index= ((sum_of_values**2)/(sum_of_squares*env.num_devices) )if sum_of_squares != 0 else 0
    
    #print(auv_positions)

    #print("the AOI is",env.AoI_all_nodes)
    average_age_over_episodes.append(np.mean(env.reward_per_step))
    average_energy_harvested_over_episodes.append(env.energy_harvested)
    jain_index_over_episodes.append(Jain_index)
    print("occurence",env.occurence)


print("This is for the average age for RL algorithm ",env.num_devices," nodes",np.mean(average_age_over_episodes))
print(" total nbr of communication RL 2d with",env.num_devices,"nodes ",np.sum(env.occurence))

print("This is the average  cummulative energy harvested for RL algorithm ",env.num_devices," nodes",np.mean(average_energy_harvested_over_episodes))
print("This is the average  Jain'fairness index for RL algorithm ",env.num_devices," nodes",np.mean(jain_index_over_episodes))
