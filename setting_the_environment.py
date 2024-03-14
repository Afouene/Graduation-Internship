import gym
from gym import spaces
import numpy as np
import pygame
def attenuation_factor(f, d, k, D, A):
        alfa_modelHOP = (0.11*(f**2/(f**2+1))
            +44*(f**2/(f**2+4100))+2.75*pow(10,-4)*pow(f,2)+
            0.003)*pow(10,-3)
        A_D_val = alfa_modelHOP * (1 - 1.93*pow(10,-5) * D)
        log_d = np.where(d < 1, 0, np.log10(200*d))
        return k * log_d + d * A_D_val + A
    
class AUVEnvironment(gym.Env):
    def __init__(self):
        super(AUVEnvironment, self).__init__()
        self.window_size = 500 #  the window size
        self.render_mode = "human"  # because we need real time visualization the render mode is human
        self.metadata = {"render_fps": 30}  
        self.auv_position = np.array([3, 3, 3]) # position of the auv at the center of network
        self.sensor_node_positions = [
            np.array([1, 1, 1]),
            np.array([5, 5, 5]),
            np.array([5, 1, 1]),
            np.array([1, 5, 5]),
            np.array([3, 3, 1])
        ]
        self.AoI_all_nodes=[1,1,1,1,1]  # we will make it not constant next time
        self.max_iterations=100

        self.AoI_max=self.max_iterations/2
        self.prev_selected_node_data = None  
        self.reward_per_step=[]
        self.action_space = spaces.MultiDiscrete([6,5,5])  #  we have 6 directions + 5 for the selection of  sensor node actions
        self.observation_space = spaces.Box(low=1, high=5, shape=(3,)) #grid 5*5*5
        self.cumulative_rewards = [0] * len(self.sensor_node_positions)
        

    def step(self, action):
        reward=0
        direction,selection_node_wet,selection_node_collect_data=action
        possible_dir=self.get_possible_directions()
        if direction in possible_dir:
            self.auv_position += np.array([
                    1 if direction == 0 else -1 if direction == 1 else 0,
                    1 if direction == 2 else -1 if direction == 3 else 0,
                    1 if direction == 4 else -1 if direction == 5 else 0
                ])
        else:
            direction=np.random.choice(possible_dir)
            self.auv_position += np.array([
                    1 if direction == 0 else -1 if direction == 1 else 0,
                    1 if direction == 2 else -1 if direction == 3 else 0,
                    1 if direction == 4 else -1 if direction == 5 else 0
                ])
        selected_sensor_node = self.sensor_node_positions[selection_node_wet]
        selected_sensor_node_collect_data=self.sensor_node_positions[selection_node_collect_data]
        received_power = self.compute_received_power(selected_sensor_node)
        reward += np.sum(received_power)
        
        #self.auv_position = np.clip(self.auv_position, 1, 5) # for auv to stay in the grid
        #selected_sensor_nodes_data = [i for i, node_pos in enumerate(self.sensor_node_positions) if self.is_in_coverage_area(node_pos)]
        d = np.linalg.norm(selected_sensor_node_collect_data - self.auv_position)
        if (d>1):
            reward -= 3

            AoI=self.update_all_Age()


        else:

            #selection_node_data = np.random.choice(selected_sensor_nodes_data)
            AoI=self.update_Age(selection_node_collect_data)
            #self.prev_selected_node_data=selection_node_data
            
        
        reward -=2*((np.sum(AoI)))/5
        self.reward_per_step.append(np.sum(AoI)/5)
        """if(np.max(AoI)==self.AoI_max):

            max_AoI_count = AoI.count(self.AoI_max)

            reward -= (max_AoI_count * self.AoI_max) *0.6"""
        
        self.cumulative_rewards[selection_node_wet] += np.sum(received_power)  
        self.max_iterations -= 1
        state = self._get_observation()
        if self.max_iterations <=0:
            done = True
        else:
            done = False
        
        return state, reward, done,{}

    def reset(self):

        self.auv_position = np.array([3, 3, 3])   
        self.max_iterations=100
        self.AoI_all_nodes=[1,1,1,1,1]
        self.cumulative_rewards = [0] * len(self.sensor_node_positions)
        return self.auv_position
    
   

    
    def _get_observation(self):
       
        return self.auv_position
    
   
    
    def compute_received_power(self, sensor_node_position):
         # Constants
        f = 500  # Frequency (KHz)
        k = 1.5  
        A = 0    
        D=100
        P_initial = 30  # Initial power (dB)

        # Distance between AUV and sensor node
        d = np.linalg.norm(sensor_node_position - self.auv_position)
        attenuation = attenuation_factor(f, d, k, D, A)  
        received_power = P_initial - attenuation
        return received_power
    def update_Age(self,node_selected_index):
        self.AoI_all_nodes[node_selected_index]=1
        for i in range(len(self.AoI_all_nodes)):
            if i != node_selected_index:  
                self.AoI_all_nodes[i] = min(self.AoI_max, self.AoI_all_nodes[i] + 1)
                #self.AoI_all_nodes[i] =  self.AoI_all_nodes[i] + 1
        return self.AoI_all_nodes
    
    def update_all_Age(self):
        for i in range(len(self.AoI_all_nodes)):
                self.AoI_all_nodes[i] = min(self.AoI_max, self.AoI_all_nodes[i] + 1)
                #self.AoI_all_nodes[i] =  self.AoI_all_nodes[i] + 1
        return self.AoI_all_nodes
    
    def get_cumulative_rewards(self):
        return self.cumulative_rewards

    def find_nearest_node_index(self):
        distances = [np.linalg.norm(sensor_node_position - self.auv_position) for sensor_node_position in self.sensor_node_positions]
        return np.argmin(distances)
    
   #  the action mask function is used so we won't get outside of the grid boundaries
    def get_possible_directions(self):
   
        possible_mvt = np.ones(6) # because we have directions
        
        if self.auv_position[0] == 1:
            possible_mvt[1] = 0  
        if self.auv_position[0] == 5:
            possible_mvt[0] = 0  
        if self.auv_position[1] == 1:
            possible_mvt[3] = 0  
        if self.auv_position[1] == 5:
            possible_mvt[2] = 0  
        if self.auv_position[2] == 1:
            possible_mvt[5] = 0  
        if self.auv_position[2] == 5:
            possible_mvt[4] = 0  
        
        possible_directions=np.where(possible_mvt==1)[0]


        return possible_directions
    
    def render(self):
        
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

    
        self.screen.fill((255, 255, 255)) 

    #  The Draw  of grid lines
        cell_size = self.window_size // 5
        for i in range(7):
            pygame.draw.line(self.screen, (100, 100, 100), (i * cell_size, 0), (i * cell_size, self.window_size), 1)
            pygame.draw.line(self.screen, (100, 100, 100), (0, i * cell_size), (self.window_size, i * cell_size), 1)

    #  The Draw of  AUV
        auv_position = self.auv_position
        auv_x = (auv_position[0]-1 ) * cell_size + cell_size // 2
        auv_y = (auv_position[1] -1) * cell_size + cell_size // 2
        pygame.draw.circle(self.screen, (0, 0, 255), (auv_x, auv_y), cell_size // 4)

    #  The Draw  of sensor nodes
        for sensor_node_pos in self.sensor_node_positions:
            node_x = (sensor_node_pos[0] -1) * cell_size + cell_size // 2
            node_y = (sensor_node_pos[1] -1) * cell_size + cell_size // 2
            pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(node_x - cell_size // 8, node_y - cell_size // 8, cell_size // 4, cell_size // 4))

        pygame.display.update()


        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.time.delay(600) 

    def _render_frame(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
