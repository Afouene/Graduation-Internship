from stable_baselines3 import PPO
from setting_the_environment import AUVEnvironment  
import numpy as np
import matplotlib.pyplot as plt

#model_path="logs/51/rl_model_9750000_steps.zip" 
#model_path="logs/53/rl_model_4860000_steps.zip" #7 nodes
#model_path='logs/58/rl_model_260000_steps.zip'
#model_path='logs/59/rl_model_670000_steps.zip' #3 nodes 2 d
#model_path='logs/56/rl_model_3055000_steps.zip'
#model_path='logs/56/rl_model_3975000_steps.zip' #5 nodes 2d

#model_path='logs/62/rl_model_2000000_steps.zip' #3 nodes 2 d
model_path="logs/67/rl_model_8720000_steps.zip" 


model = PPO.load(model_path)


env = AUVEnvironment()

num_episodes = 1
average_age_over_episodes= []
average_energy_harvested_over_episodes=[]
jain_index_over_episodes = []  

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs,deterministic=True)  
        obs, reward, done, _ = env.step(action)  
        total_reward += reward

        env.render()
        print("auv position",env.auv_position)
        print("action",action)
    
    sum_of_squares = sum(x**2 for x in env.occurence)
    sum_of_values = sum(env.occurence)

    Jain_index= ((sum_of_values**2)/(sum_of_squares*env.num_devices) )if sum_of_squares != 0 else 0

        
    #print("the AOI is",env.AoI_all_nodes)
    average_age_over_episodes.append(np.mean(env.reward_per_step))
    average_energy_harvested_over_episodes.append(env.energy_harvested)
    jain_index_over_episodes.append(Jain_index)
    print("occurence",env.occurence)
   
    


    #print("This is the power transfered to nodes",env.cumulative_rewards)
    #print("This the average reward",abs(np.mean((env.reward_per_step))))
print("This is for the average age for RL algorithm ",env.num_devices," nodes",np.mean(average_age_over_episodes))

print("This is the average  cummulative energy harvested for RL algorithm ",env.num_devices," nodes",np.mean(average_energy_harvested_over_episodes))
print("This is the average  Jain'fairness index for RL algorithm ",env.num_devices," nodes",np.mean(jain_index_over_episodes))

