import matplotlib.pyplot as plt
import numpy as np
from setting_the_environment import AUVEnvironment

total_rewards_over_episodes = []
num_episodes = 50

for episode in range(num_episodes):
    done=False
    total_reward = 0
    env = AUVEnvironment()
    while not done : 
        auv_positions = [env.auv_position]      # Store AUV positions for plotting
        action = env.action_space.sample()
        direction,selected_node=action
        next_state, reward, done ,_= env.step(action)
        total_reward += reward
        env.render()
        """print("Step: Reward:", reward)
        print("pos of auv",env.auv_position)
        print("The action of the direction",direction)
        print("node selected",selected_node)
        # Store AUV position for plotting
        auv_positions.append(env.auv_position)"""

    total_rewards_over_episodes.append(total_reward)
    #cumulative_rewards = env.get_cumulative_rewards()
    print("total reward for the episode is  ", total_reward)

    env.close()
mean_reward = np.mean(total_rewards_over_episodes)
std_deviation = np.std(total_rewards_over_episodes)

print("Mean total reward over", num_episodes, "episodes:", mean_reward)
print("Standard deviation of total rewards over", num_episodes, "episodes:", std_deviation)

    # Print cumulative rewards for each node 
"""for i, reward in enumerate(cumulative_rewards):
        print(f"Cumulative reward for node {i}: {reward}")
    print("total reward is ", total_reward)"""
plt.plot(range(1, num_episodes + 1), total_rewards_over_episodes, marker='o', linestyle='None')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward after 20 Steps for Each Episode Random selection')
plt.grid(True)
plt.show()

