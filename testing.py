import gym
from setting_the_environment import AUVEnvironment

env=AUVEnvironment()
state=env.reset()
num_steps=20
total_reward=0
for _ in range(num_steps):
    action=env.action_space.sample()
    next_state,reward,_=env.step(action)
    total_reward+=reward
    print("Step:", _, "Reward:", reward)
    
    # If the episode is done, reset the environment
    """if done:
        print("Episode finished after", _+1, "steps")
        state = env.reset()
        break
     """   
# Print total reward obtained in the episode
print("Total Reward:", total_reward)

# Close the environment
env.close()

