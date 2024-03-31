<<<<<<< HEAD
from stable_baselines3 import PPO
from environment import AUVEnvironment  
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
#model_path = "training/PPO/100000.zip"
#model_path="logs/models/rl_model_150000_steps.zip"
#model_path="logs/models_mu_eq_2/rl_model_200000_steps.zip"
#model_path='logs/models_mu_eq_10/rl_model_200000_steps.zip'
#model_path='logs/models_mu_eq_0.2/rl_model_175000_steps.zip'
#model_path='logs/models_mu_eq_0.5/rl_model_130000_steps.zip'
#model_path="logs/models_after_changing_dimensions/rl_model_90000_steps.zip"
#model_path="logs/models_new_formula_harvested_power/rl_model_100000_steps.zip"
#model_path="logs/models_new_formula_harvested_power_version_2/rl_model_100000_steps.zip"
#model_path="logs/models_1105/rl_model_40000_steps.zip"
#model_path="logs/models_1351/rl_model_115000_steps.zip"
#model_path='logs/models_1449/rl_model_100000_steps.zip'
model_path="logs/models_1204_28/rl_model_150000_steps.zip"
model = PPO.load(model_path)


env = AUVEnvironment()

num_episodes = 100
average_age_over_episodes= []
average_power_transfer_over_episodes=[]
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs,deterministic=False)  
        obs, reward, done, _ = env.step(action)  
        total_reward += reward

        env.render()
        

    """
        print("action",action)
        #print("dis",np.linalg.norm(env.sensor_node_positions[action[1]] - env.auv_position))
    print("the aoi of each node",env.AoI_all_nodes)
    print("the average aoi is ",np.sum(env.AoI_all_nodes)/5)
    print(f"Episode {episode + 1}: Total reward = {total_reward}")
    print('total rewards for all nodes',env.cumulative_rewards)"""
    #print("the AOI is",env.AoI_all_nodes)
    average_age_over_episodes.append(np.mean(env.reward_per_step))
    print('coofafa',env.cumulative_rewards)
    average_power_transfer_over_episodes.append(np.sum(env.cumulative_rewards))

    #print("This is the power transfered to nodes",env.cumulative_rewards)
    #print("This the average reward",abs(np.mean((env.reward_per_step))))

print("This is for the average age",np.mean(average_age_over_episodes))
print("This is the average power transfered",np.mean(average_power_transfer_over_episodes))
=======
from stable_baselines3 import PPO
from setting_the_environment import AUVEnvironment  
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
#model_path = "training/PPO/100000.zip"
#model_path="logs/models/rl_model_150000_steps.zip"
#model_path="logs/models_mu_eq_2/rl_model_200000_steps.zip"
#model_path='logs/models_mu_eq_10/rl_model_200000_steps.zip'
#model_path='logs/models_mu_eq_0.2/rl_model_175000_steps.zip'
#model_path='logs/models_mu_eq_0.5/rl_model_130000_steps.zip'
#model_path="logs/models_after_changing_dimensions/rl_model_90000_steps.zip"
#model_path="logs/models_new_formula_harvested_power/rl_model_100000_steps.zip"
#model_path="logs/models_new_formula_harvested_power_version_2/rl_model_100000_steps.zip"
model_path="logs/models_constrained_energy-10/rl_model_150000_steps.zip"

model = PPO.load(model_path)


env = AUVEnvironment()

num_episodes = 1000
average_age_over_episodes= []
average_power_transfer_over_episodes=[]
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=False)  
        obs, reward, done, _ = env.step(action)  
        total_reward += reward

        #env.render()
        

    """
        print("action",action)
        #print("dis",np.linalg.norm(env.sensor_node_positions[action[1]] - env.auv_position))
    print("the aoi of each node",env.AoI_all_nodes)
    print("the average aoi is ",np.sum(env.AoI_all_nodes)/5)
    print(f"Episode {episode + 1}: Total reward = {total_reward}")
    print('total rewards for all nodes',env.cumulative_rewards)"""
    #print("the AOI is",env.AoI_all_nodes)
    average_age_over_episodes.append(np.mean(env.reward_per_step))
    print('Power transfered',env.cumulative_rewards)
    average_power_transfer_over_episodes.append(np.sum(env.cumulative_rewards))

    #print("This is the power transfered to nodes",env.cumulative_rewards)
    #print("This the average reward",abs(np.mean((env.reward_per_step))))

print("This is for the average age",np.mean(average_age_over_episodes))
print("This is the average  cummulative power transfered",np.mean(average_power_transfer_over_episodes))
>>>>>>> 528c9a91fed089556c00275ae858532c11cabcc7
env.close()