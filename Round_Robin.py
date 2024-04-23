


from setting_the_environment import AUVEnvironment  


import numpy as np



class RoundRobinAgent:
    def __init__(self, num_devices):
        self.num_devices = num_devices
        self.current_node_index = 0  # Start from the first node

    def select_action(self):
        # Randomly select direction
        direction = np.random.randint(6)  # Assuming 6 possible directions
        # Select the current node for both power transmission and data collection
        selected_node_index = self.current_node_index
        action = (direction, selected_node_index, selected_node_index)
        # Move to the next node in a cyclic manner
        self.current_node_index = (self.current_node_index + 1) % self.num_devices
        return action

if __name__ == "__main__":
    total_rewards_over_episodes_rr = []
    average_age_over_episodes_rr = []
    average_energy_harvested_over_episodes_rr = []
    num_episodes = 100

    for episode in range(num_episodes):
        done = False
        env = AUVEnvironment()
        agent = RoundRobinAgent(env.num_devices)

        total_reward = 0
        while not done:
            action = agent.select_action()
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
        
        #print("occurence",env.occurence)


        average_age_over_episodes_rr.append(np.mean(env.reward_per_step))
        average_energy_harvested_over_episodes_rr.append(env.energy_harvested)

    print("Average age for Round Robin with", env.num_devices, "nodes:", np.mean(average_age_over_episodes_rr))
    print("Average cumulative energy harvested for Round Robin with", env.num_devices, "nodes:", np.mean(average_energy_harvested_over_episodes_rr))