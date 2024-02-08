import gym
from gym import spaces
import numpy as np

def attenuation_factor(f, d, k, D, A):
        alfa_modelHOP = (0.11*(f**2/(f**2+1))
            +44*(f**2/(f**2+4100))+2.75*pow(10,-4)*pow(f,2)+
            0.003)*pow(10,-3)
        A_D_val = alfa_modelHOP * (1 - 1.93*pow(10,-5) * D)
        log_d = np.where(d < 1, 0, np.log10(d))
        return k * log_d + d * A_D_val + A
    
class AUVEnvironment(gym.Env):
    def __init__(self):
        super(AUVEnvironment, self).__init__()
        self.auv_position = np.array([3, 3, 3])
        self.sensor_node_positions = [
            np.array([1, 1, 1]),
            np.array([5, 5, 5]),
            np.array([2, 5, 5]),
            np.array([5, 2, 2]),
            np.array([4, 4, 4])
        ]
        self.action_space = spaces.Tuple((spaces.Discrete(6), spaces.Discrete(5)))  # 6 directions + 5 select sensor node actions
        self.observation_space = spaces.Box(low=1, high=5, shape=(3,)) #grid 5*5*5
        self.max_steps=50
        

    def step(self, action):
        reward=0
        direction,selection_node=action
        
            # Move AUV in one of the six directions
        self.auv_position += np.array([
                1 if direction == 0 else -1 if direction == 1 else 0,
                1 if direction == 2 else -1 if direction == 3 else 0,
                1 if direction == 4 else -1 if direction == 5 else 0
            ])
        self.auv_position = np.clip(self.auv_position, -5, 5)
        
        selected_sensor_node = self.sensor_node_positions[selection_node]
        received_power = self.compute_received_power(selected_sensor_node)
            # Compute reward
        reward = np.sum(received_power)

        # Update state
        state = self._get_observation()
        """if self.current_step >= self.max_steps:
            done = True
        else:
            done = False
        """
        return state, reward, {}

    def reset(self):
        # Reset AUV position to center
        self.auv_position = np.array([0, 0, 0])
        # Return initial state
        return self._get_observation()

    def render(self, mode='human'):
        # Implement visualization if needed
        pass

    def _get_observation(self):
        received_powers = np.array([self.compute_received_power(sensor_node_pos) for sensor_node_pos in self.sensor_node_positions])

    # Construct the observation array with AUV position and received powers
        observation = np.concatenate((self.auv_position, received_powers), axis=0)
        return observation
    
   
    
    def compute_received_power(self, sensor_node_position):
         # Constants
        f = 100  # Frequency (KHz)
        k = 1.5  # Constant for attenuation calculation
        A = 0    # Constant for attenuation calculation
        D=100
        P_initial = 10  # Initial power (dB)

        # Distance between AUV and sensor node
        d = np.linalg.norm(sensor_node_position - self.auv_position)
        attenuation = attenuation_factor(f, d, k, D, A)  
        received_power = P_initial - attenuation
        return received_power


    
    def close(self):
        # Clean up resources if needed
        pass
