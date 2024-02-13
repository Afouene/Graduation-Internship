'''import gym
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
'''
from setting_the_environment import AUVEnvironment

env = AUVEnvironment()

# Store AUV positions for plotting
auv_positions = [env.auv_position]

num_steps = 20
total_reward = 0
for _ in range(num_steps):
    action = env.action_space.sample()
    direction,selected_node=action
    next_state, reward, _ = env.step(action)
    total_reward += reward
    env.render()
    print("Step:", _, "Reward:", reward)
    print("pos of auv",env.auv_position)
    print("The action of the direction",direction)
    print("node selected",selected_node)
    # Store AUV position for plotting
    auv_positions.append(env.auv_position)

cumulative_rewards = env.get_cumulative_rewards()

# Print cumulative rewards for each node 
for i, reward in enumerate(cumulative_rewards):
    print(f"Cumulative reward for node {i}: {reward}")


env.close()
