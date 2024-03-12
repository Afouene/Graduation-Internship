from stable_baselines3 import PPO
from setting_the_environment import AUVEnvironment  
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model_path = "training/PPO/100000.zip"
model = PPO.load(model_path)

# Create the environment
env = AUVEnvironment()

# Test the trained model for 5 episodes
num_episodes = 100
reward_over_episodes= []
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=False)  
        obs, reward, done, _ = env.step(action)  
        total_reward += reward

        #env.render()
    """print("AUV position:", env.auv_position)
        print("action",action)
        #print("dis",np.linalg.norm(env.sensor_node_positions[action[1]] - env.auv_position))
    print("the aoi of each node",env.AoI_all_nodes)
    print("the average aoi is ",np.sum(env.AoI_all_nodes)/5)
    print(f"Episode {episode + 1}: Total reward = {total_reward}")
    print('total rewards for all nodes',env.cumulative_rewards)"""
    print("the AOI is",env.AoI_all_nodes)

    reward_over_episodes.append(np.mean(env.reward_per_step))
    #print("This the average reward",abs(np.mean((env.reward_per_step))))

print("This is for the average reward",np.mean(reward_over_episodes))
"""
plt.plot(range(1, num_episodes + 1), reward_over_episodes)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Average AoI per Episode')
plt.grid(True)
plt.show()
# Close the environment
"""
env.close()