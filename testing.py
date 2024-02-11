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
import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    print("direction",direction)
    print("node selected",selected_node)
    # Store AUV position for plotting
    auv_positions.append(env.auv_position)

"""auv_positions = np.array(auv_positions)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(auv_positions[:, 0], auv_positions[:, 1], auv_positions[:, 2], marker='o', label='AUV')
sensor_node_positions = np.array(env.sensor_node_positions)
ax.scatter(sensor_node_positions[:, 0], sensor_node_positions[:, 1], sensor_node_positions[:, 2], color='r', marker='s', label='Sensor Nodes')
ax.set_title('AUV Trajectory and Sensor Node Positions')
ax.set_xlabel('X position')
ax.set_ylabel('Y position')
ax.set_zlabel('Z position')
ax.legend()
plt.show()

# Print total reward obtained in the episode
print("Total Reward:", total_reward)
print("nbm actions",env.action_space)
"""
# Close the environment
env.close()
