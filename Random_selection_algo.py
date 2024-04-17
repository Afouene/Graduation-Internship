import matplotlib.pyplot as plt
import numpy as np
from setting_the_environment import AUVEnvironment

total_rewards_over_episodes = []
num_episodes = 100
average_age_over_episodes= []
average_energy_harvested_over_episodes=[]
for episode in range(num_episodes):
    done=False
    total_reward = 0
    env = AUVEnvironment()
    while not done : 
        auv_positions = [env.auv_position]      # Store AUV positions for plotting
        action = env.action_space.sample()
        direction,selected_node,selected_node_data=action
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
    average_energy_harvested_over_episodes.append(env.energy_harvested)
    #print("This is the power transfered to nodes",env.cumulative_rewards)

    #cumulative_rewards = env.get_cumulative_rewards()



print("This is for the average age for RW  10 nodes ",np.mean(average_age_over_episodes))
print("This is the average  cummulative energy harvested for RW 10 nodes",np.mean(average_energy_harvested_over_episodes))


