
import matplotlib.pyplot as plt
import numpy as np
from setting_the_environment import AUVEnvironment
import gym
total_rewards_over_episodes = []
num_episodes = 100
average_age_over_episodes= []
average_energy_harvested_over_episodes=[]
jain_index_over_episodes = []  

for episode in range(num_episodes):
    done=False
    env = AUVEnvironment()
    total_reward = 0
    while not done : 
        action = env.action_space.sample()
        direction,selected_node,selection_node_data=action
        next_state, reward, done ,_= env.step(action)
        total_reward += reward  
      
        
        #env.render()
        #print("auv position",env.auv_position)
        
    average_age_over_episodes.append(np.mean(env.reward_per_step))
    #print("This is aoi",env.reward_per_step)
    #print("This is the power transfered to nodes",env.cumulative_rewards)
    average_energy_harvested_over_episodes.append(env.energy_harvested)
    #print("occurence",env.occurence)
    sum_of_squares = sum(x**2 for x in env.occurence)
    sum_of_values = sum(env.occurence)

    Jain_index= ((sum_of_values**2)/(sum_of_squares*env.num_devices) )
    jain_index_over_episodes.append(Jain_index)




   

print("This is for the average age for RW ",env.num_devices,"  nodes",np.mean(average_age_over_episodes))
print("This is the average  cummulative energy harvested for RW to",env.num_devices," nodes ",np.mean(average_energy_harvested_over_episodes))
print("This is the average  Jain'fairness index for RW algorithm ",env.num_devices," nodes",np.mean(jain_index_over_episodes))

""""
plt.plot(range(1, num_episodes + 1), total_rewards_over_episodes)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.grid(True)
plt.show()"""


