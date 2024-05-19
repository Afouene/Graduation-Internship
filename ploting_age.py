import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# AUV position
auv_position = np.array([5, 5, 2])

# Sensor node positions
sensor_node_positions = np.array([
    [1, 1, 1],
    [8, 8, 2],
    [9, 1, 3],
    [1, 6, 4],
    [8, 5, 3],
    [6, 2, 2],
    [4, 4, 3],
    [3, 7, 2],
    [5, 8, 4],
    [6, 9, 3],
])

# Plotting the positions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(auv_position[0], auv_position[1], auv_position[2], color='b', label='AUV')

for pos in sensor_node_positions:
    ax.scatter(pos[0], pos[1], pos[2], color='r', label='Sensor Node')

# Adding annotations
ax.text(auv_position[0], auv_position[1], auv_position[2], 'AUV', color='b')

for i, pos in enumerate(sensor_node_positions):
    ax.text(pos[0], pos[1], pos[2], f'Node {i+1}', color='r')

# Setting labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Adding legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

# Show plot
plt.show()
