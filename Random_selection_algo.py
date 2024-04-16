
import matplotlib.pyplot as plt
import numpy as np
from setting_the_environment import AUVEnvironment
import gym
total_rewards_over_episodes = []
num_episodes = 100
average_age_over_episodes= []
average_energy_harvested_over_episodes=[]
for episode in range(num_episodes):
    done=False
    env = AUVEnvironment()

    total_reward = 0
    while not done : 
        auv_positions = [env.auv_position]      # Store AUV positions for plotting
        action = env.action_space.sample()
        direction,selected_node,selection_node_data=action
        next_state, reward, done ,_= env.step(action)
        total_reward += reward  
      
        
        #env.render()
        #print("auv position",env.auv_position)
        """print("aoi,",env.AoI_all_nodes)
        print("THIS IS AUV POSition",env.auv_position)
        print("reward",reward)
        print("The action of the direction",direction)
        print("node selected",selected_node)
        # Store AUV position for plotting
        auv_positions.append(env.auv_position)"""
    average_age_over_episodes.append(np.mean(env.reward_per_step))
    #print("This is aoi",env.reward_per_step)
    #print("This is the power transfered to nodes",env.cumulative_rewards)
    average_energy_harvested_over_episodes.append(env.energy_harvested)
    print("occurence",env.occurence)
    """print("number of times the auv chosed a node",env.t)
    print("number of times the auv chosed a node correctly",env.t1)
    print("number of times the auv couldnt chose a node",env.f)"""




    #cumulative_rewards = env.get_cumulative_rewards()
    """print("total reward for the episode is  ", total_reward)
    print("the aoi is",env.AoI_all_nodes)
    env.close()"""
"""mean_reward = np.mean(total_rewards_over_episodes)
std_deviation = np.std(total_rewards_over_episodes)"""

print("This is for the average age",np.mean(average_age_over_episodes))
print("This is the average  cummulative energy harvested",np.mean(average_energy_harvested_over_episodes))
""""
plt.plot(range(1, num_episodes + 1), total_rewards_over_episodes)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward per Episode')
plt.grid(True)
plt.show()"""


