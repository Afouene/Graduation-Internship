from stable_baselines3 import PPO
from setting_the_environment import AUVEnvironment  
import numpy as np

# Load the pre-trained model
model_path = "training/PPO/200000.zip"
model = PPO.load(model_path)

# Create the environment
env = AUVEnvironment()

# Test the trained model for 5 episodes
num_episodes = 1
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=False)  
        obs, reward, done, _ = env.step(action)  
        total_reward += reward
        env.render()
       
        print("AUV position:", env.auv_position)
        print("action",action)
        
        #print("dis",np.linalg.norm(env.sensor_node_positions[action[1]] - env.auv_position))
    print(f"Episode {episode + 1}: Total reward = {total_reward}")
    print('total rewards for all nodes',env.cumulative_rewards)
# Close the environment
env.close()
