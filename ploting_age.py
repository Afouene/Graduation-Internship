from stable_baselines3 import PPO
from setting_the_environment import AUVEnvironment  
import numpy as np
import matplotlib.pyplot as plt

model_path="logs/43/rl_model_550000_steps.zip"

model = PPO.load(model_path)


env = AUVEnvironment()
env_random = AUVEnvironment()

obs_rl = env.reset()
done_rl = False
total_reward = 0
while not done_rl:
        action, _ = model.predict(obs_rl,deterministic=False)  
        obs_rl, reward_rl, done_rl, _ = env.step(action)  
        total_reward += reward_rl

obs_random = env_random.reset()
done_random = False
total_reward_random = 0
while not done_random: 
        action_random = env_random.action_space.sample()
        obs_random, reward_random, done_random, _ = env_random.step(action_random)  
        total_reward_random += reward_random


plt.plot(np.arange(1, 100 + 1), env.reward_per_step, label="Age per Step for RL")
plt.plot(np.arange(1, 100 + 1), env_random.reward_per_step, label="Age per Step for RW")

plt.xlabel("step")
plt.ylabel(" Average Age per Step")
plt.title("Average Age per Step for 1 Episode")
plt.legend()
plt.show()