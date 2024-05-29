import numpy as np
from new_env2d import AUVEnvironment  

class GreedyAlgorithm:
    def __init__(self, auv_position, sensor_node_positions):
        self.auv_position = auv_position
        self.sensor_node_positions = sensor_node_positions
        self.center_of_gravity = np.mean(self.sensor_node_positions, axis=0)
    def calculate_distance_to_center_of_gravity(self, new_position):
        distance = np.linalg.norm(self.center_of_gravity - new_position)
        return distance
    def find_best_direction_to_center_of_gravity(self):
        best_direction = None
        min_distance = float('inf')
        for direction in range(6):
            new_position = self.auv_position + np.array([
                1 if direction == 0 else -1 if direction == 1 else 0,
                1 if direction == 2 else -1 if direction == 3 else 0,
                1 if direction == 4 else -1 if direction == 5 else 0
            ])
            distance = self.calculate_distance_to_center_of_gravity(new_position)
            if distance < min_distance:
                min_distance = distance
                best_direction = direction
        return best_direction
    
    def find_closest_node_to_charge(self):
        distances = [np.linalg.norm(node_position - self.auv_position) for node_position in self.sensor_node_positions]
        closest_node_index = np.argmin(distances)
        return closest_node_index

    def select_action(self):
        direction=self.find_best_direction_to_center_of_gravity() 
        if(np.linalg.norm(self.center_of_gravity - self.auv_position)<=1):
            node_selected=np.random.randint(len(self.sensor_node_positions))

        else :
            node_selected=self.find_closest_node_to_charge()
        action=(direction,node_selected)
        return action
if __name__ == "__main__":
    total_rewards_over_episodes_ga = []
    average_age_over_episodes_ga = []
    average_energy_harvested_over_episodes_ga= []
    average_total_comm=[]
    jain_index_over_episodes=[]
    num_episodes = 100

    for episode in range(num_episodes):
        done = False
        env = AUVEnvironment()
        agent = GreedyAlgorithm(env.auv_position,env.sensor_node_positions)

        total_reward = 0
        while not done:
            action = agent.select_action()
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            #env.render()
        sum_of_squares = sum(x**2 for x in env.occurence)
        sum_of_values = sum(env.occurence)

        Jain_index= ((sum_of_values**2)/(sum_of_squares*env.num_devices) )if sum_of_squares != 0 else 0
        jain_index_over_episodes.append(Jain_index)

        average_total_comm.append(np.sum(env.occurence))
        average_age_over_episodes_ga.append(np.mean(env.reward_per_step))
        average_energy_harvested_over_episodes_ga.append(env.energy_harvested)

    print("Average age for Greedy Algorithm with", env.num_devices, "nodes:", np.mean(average_age_over_episodes_ga))
    print("Average cumulative energy harvested for Greedy Algorithm with", env.num_devices, "nodes:", np.mean(average_energy_harvested_over_episodes_ga))
    print("This is the average  Jain'fairness index for RW algorithm ",env.num_devices," nodes",np.mean(jain_index_over_episodes))
    print("Average total nbr of communication  RW with",env.num_devices,"nodes ",np.mean(average_total_comm))