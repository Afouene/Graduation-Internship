from stable_baselines3 import A2C
from setting_the_environment import AUVEnvironment  
import numpy as np

# Load the pre-trained model
model_path = "training/saved_models/A2C_model_version1_ent"
model = A2C.load(model_path)

# Create the environment
env = AUVEnvironment()

# Test the trained model for 5 episodes
num_episodes = 1
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)  # Predict action using the model
        obs, reward, done, _ = env.step(action)  # Take action in the environment
        total_reward += reward
        # Optionally, render the environment
        env.render()
        # Print AUV position
        print("AUV position:", env.auv_position)
        print("action",action)
        #print("reward",reward)
        #print("dis",np.linalg.norm(env.sensor_node_positions[action[1]] - env.auv_position))
    print(f"Episode {episode + 1}: Total reward = {total_reward}")

# Close the environment
env.close()
