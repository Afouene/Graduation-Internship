import matplotlib.pyplot as plt
import numpy as np
from setting_the_environment import AUVEnvironment

total_rewards_over_episodes = []
num_episodes = 50

for episode in range(num_episodes):
    done = False
    total_reward = 0
    env = AUVEnvironment()
    while not done:
        action=env.action_space.sample()
        nearest_node_index = env.find_nearest_node_index() # the index of sensor node that is the closest to the auv
        action = (action[0], nearest_node_index)
        
        # Take a step in the environment
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        env.render()
        print("auvpos",env.auv_position)

    total_rewards_over_episodes.append(total_reward)
    print("Total reward for the episode is:", total_reward)

    env.close()

mean_reward = np.mean(total_rewards_over_episodes)
std_deviation = np.std(total_rewards_over_episodes)

print("Mean total reward over", num_episodes, "episodes:", mean_reward)
print("Standard deviation of total rewards over", num_episodes, "episodes:", std_deviation)

# Plot total rewards over episodes
plt.plot(range(1, num_episodes + 1), total_rewards_over_episodes, marker='o', linestyle='None')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Reward after 20 Steps for Each Episode')
plt.grid(True)
plt.show()
