import gym
from gym import spaces
import numpy as np
import pygame
def attenuation_factor(f, d, k, D, A):
        alfa_modelHOP = (0.11*(f**2/(f**2+1))
            +44*(f**2/(f**2+4100))+2.75*pow(10,-4)*pow(f,2)+
            0.003)*pow(10,-3)
        A_D_val = alfa_modelHOP * (1 - 1.93*pow(10,-5) * D)
        log_d = np.where(d < 1, 0, np.log10(100*(d+1)))
        return k * log_d + d * A_D_val + A
    
class AUVEnvironment(gym.Env):
    def __init__(self):
        super(AUVEnvironment, self).__init__()
        self.window_size = 500 #  the window size
        self.render_mode = "human"  # because we need real time visualization the render mode is human
        self.metadata = {"render_fps": 30}  
        self.screen = None
        self.clock = None
        self.auv_position = np.array([3, 3, 3]) # position of the auv at the center of network
        self.sensor_node_positions = [
            np.array([1, 1, 1]),
            np.array([5, 5, 5]),
            np.array([2, 5, 5]),
            np.array([5, 2, 2]),
            np.array([4, 4, 4])
        ]
        self.action_space = spaces.Tuple((spaces.Discrete(6), spaces.Discrete(5)))  # 6 directions + 5 select sensor node actions
        self.observation_space = spaces.Box(low=1, high=5, shape=(3,)) #grid 5*5*5
        

    def step(self, action):
        reward=0
        direction,selection_node=action
        
            # Move AUV in one of the six directions
        self.auv_position += np.array([
                1 if direction == 0 else -1 if direction == 1 else 0,
                1 if direction == 2 else -1 if direction == 3 else 0,
                1 if direction == 4 else -1 if direction == 5 else 0
            ])
        self.auv_position = np.clip(self.auv_position, 1, 5) # for auv to stay in the grid
        
        selected_sensor_node = self.sensor_node_positions[selection_node]
        received_power = self.compute_received_power(selected_sensor_node)
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

        self.auv_position = np.array([3, 3, 3])   # Reset AUV position to center

        return self._get_observation()


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
    def render(self):
        
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

        # Draw your environment elements here
        self.screen.fill((255, 255, 255))

    # Draw grid lines
        cell_size = self.window_size // 5
        for i in range(6):
            pygame.draw.line(self.screen, (100, 100, 100), (i * cell_size, 0), (i * cell_size, self.window_size), 1)
            pygame.draw.line(self.screen, (100, 100, 100), (0, i * cell_size), (self.window_size, i * cell_size), 1)

    # Draw AUV
        auv_position = self.auv_position
        auv_x = (auv_position[0] - 1) * cell_size + cell_size // 2
        auv_y = (auv_position[1] - 1) * cell_size + cell_size // 2
        pygame.draw.circle(self.screen, (0, 0, 255), (auv_x, auv_y), cell_size // 4)

    # Draw sensor nodes
        for sensor_node_pos in self.sensor_node_positions:
            node_x = (sensor_node_pos[0] - 1) * cell_size + cell_size // 2
            node_y = (sensor_node_pos[1] - 1) * cell_size + cell_size // 2
            pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(node_x - cell_size // 8, node_y - cell_size // 8, cell_size // 4, cell_size // 4))

        pygame.display.update()


        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.time.delay(600) 

    def _render_frame(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Draw your environment elements here

        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
