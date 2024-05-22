import gym
from gym import spaces
import numpy as np
import pygame
from scipy.optimize import bisect

c1 = 0.03366340416928765

def signal_to_noise_ratio(SL, TL, NL):
    return SL - TL - NL

def Acoustic_source_level(P_elec, elec_acous_conv_eff, DI):
    return 170.8 + 10 * np.log10(P_elec) + 10 * np.log10(elec_acous_conv_eff) + DI

def Transmission_Loss(f, k, r):
    alfa_THorp = (0.11 * (f**2 / (f**2 + 1)) + 44 * (f**2 / (f**2 + 4100)) + 2.75e-4 * f**2 + 0.003) * 1e-3
    log_r = np.where(r < 1, 0, np.log10(r))
    return k * log_r + r * alfa_THorp

def Power_harvested(n, RL, RVS, Rp):
    p = 10**(RL / 20)  # acoustic pressure p on the hydrophone
    V_ind = p * 10**(RVS / 20)  # induced voltage
    P_available = n * (V_ind**2) / (4 * Rp)
    P_har = 0.7 * P_available
    return P_har

def snr_needed_for_transmission_data(system_throughput, Bandiwdth):
    return 2**(system_throughput / Bandiwdth) - 1




def f(x):
    c2 = 4 / 3000
    term1 = np.log10(x / (1 - x))
    term2 = np.log10(2**(c2 / (1 - x)) - 1)
    return term1 - term2

def h(d,f):
    log_term = 0 if d < 1 else 1.5 * np.log10(d)
    c3=(0.11*(f**2/(f**2+1))
            +44*(f**2/(f**2+4100))+2.75*pow(10,-4)*pow(f,2)+
            0.003)*(10**(-3))
    return 0.2 * (log_term + d * c3) - np.log10(c1)

def find_beta(y):
    def equation(x):
        return f(x) - y
    try:
        beta = bisect(equation, 0.1, 0.9)
        return beta
    except ValueError:
        return None

class AUVEnvironment(gym.Env):
    def __init__(self):
        super(AUVEnvironment, self).__init__()
        self.window_size = 500  # the window size
        self.render_mode = "human"  # because we need real-time visualization, the render mode is human
        self.metadata = {"render_fps": 30}
        self.auv_position = np.array([5, 5, 2])  # position of the AUV at the center of the network

        self.sensor_node_positions = [
            np.array([1, 1, 1]),
            np.array([8, 8, 2]),
            np.array([9, 1, 3]),
            np.array([1, 6, 4]),
            np.array([8, 5, 3]),
            np.array([6, 2, 2]),
            np.array([4, 4, 3]),
            #np.array([3, 7, 2]),
            #np.array([5, 8, 4]),
            #np.array([6, 9, 3]),
        ]

        self.num_devices = 7

        self.AoI_all_nodes = [1] * self.num_devices
        self.max_iterations = 100
        self.center_of_gravity = np.mean(self.sensor_node_positions, axis=0)

        self.AoI_max = self.max_iterations / 2
        self.reward_per_step = []
        self.action_space = spaces.MultiDiscrete([6, self.num_devices])  # we have 6 directions + 10 for the selection of sensor node actions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_devices*2+3 ,))
        self.energy_stored = [0] * self.num_devices
        self.energy_harvested = 0
        self.t = 0
        self.f = 0
        self.n = 0
        self.occurence = [0] * self.num_devices
        self.visited = [0] * self.num_devices

    def step(self, action):
        reward = 0
        direction, selection_node_wet = action
        possible_dir = self.get_possible_directions()
        if direction in possible_dir:
            self.auv_position += np.array([
                1 if direction == 0 else -1 if direction == 1 else 0,
                1 if direction == 2 else -1 if direction == 3 else 0,
                1 if direction == 4 else -1 if direction == 5 else 0
            ])
        else:
            reward -= 10

        selected_sensor_node = self.sensor_node_positions[selection_node_wet]
        r = np.linalg.norm(selected_sensor_node - self.auv_position)
        y = h(100*r,f=40)
        beta = find_beta(y)
        if beta is None or beta < 0.1 or beta > 0.9:
            if(self.energy_stored[selection_node_wet] > self.energy_required_for_trans(r,0)):
                beta=0
            else:
                beta=1
        if(beta !=0 and beta!=1):
            harvested_energy = self.compute_harvested_energy(r,beta+0.0001)
        else :
            harvested_energy = self.compute_harvested_energy(r,beta)

        self.energy_stored[selection_node_wet] += harvested_energy
        self.energy_harvested += harvested_energy

        if beta !=1:
            self.occurence[selection_node_wet] += 1
            AoI = self.update_Age(selection_node_wet)
            self.energy_stored[selection_node_wet] -= self.energy_required_for_trans(r,beta)
            if(beta==0):
                reward +=2
            if self.occurence[selection_node_wet] > 14:
                reward -= 10

        else:
            AoI = self.update_all_Age()
            reward -= 4
        sum_of_squares = sum(x**2 for x in self.occurence)
        sum_of_values = sum(self.occurence)
        Jain_index = ((sum_of_values**2) / (sum_of_squares * self.num_devices)) if sum_of_squares != 0 else 0

        num_zeros = sum(1 for x in self.occurence if x == 0)
        reward -= (1 - Jain_index) * (np.sum(AoI) / self.num_devices) + num_zeros

        self.reward_per_step.append(np.sum(AoI) / self.num_devices)
        self.max_iterations -= 1
        state = self._get_observation()
        done = self.max_iterations <= 0

        return state, reward, done, {}

    def reset(self):
        self.auv_position = np.array([5, 5, 2])
        self.max_iterations = 100
        self.AoI_all_nodes = [1] * self.num_devices
        self.energy_stored = [0] * self.num_devices
        self.energy_harvested = 0
        self.occurence = [0] * self.num_devices
        self.t = 0
        self.f = 0
        self.n = 0
        return np.hstack((self.auv_position, self.AoI_all_nodes, self.energy_stored))

    def _get_observation(self):
        return np.hstack((self.auv_position, self.AoI_all_nodes, self.energy_stored))

    def seed(self, seed=None):
        pass

    def compute_harvested_energy(self,r,beta):
        SL=Acoustic_source_level(2000,0.5,20)
        #avg_distance=0.5*(self.auv_position+self.prev_auv_position)
        #print("auv pos",self.auv_position," sensor",sensor_node_position)
        AL=Transmission_Loss(40,1.5,100*r)
        NL=30
        RVS=-150
        Rp=125
        duration=25 *beta
        RL=signal_to_noise_ratio(SL,AL,NL)
        P_harvested=Power_harvested(2,RL,RVS,Rp)

        energy_harvested=P_harvested*duration
        return energy_harvested

    def energy_required_for_trans(self,r,beta):
        if beta==1:
            snr = 0
    

        else :
            snr = snr_needed_for_transmission_data(100 / (25 * (1 - beta)), 3000)

        AL=Transmission_Loss(40,1.5,100*r)
        NL=30
        power_for_transmission=snr*(10**(AL/10))*(10**(NL/10))
        duration=25*(1-beta)
        energy_for_tranmission=power_for_transmission*duration

        return energy_for_tranmission

    def indices_state_for_transmitting(self,beta):
        E_n = [0] * self.num_devices
        e_value = [0] * self.num_devices
        for indice in range(len(self.sensor_node_positions)):
            r = np.linalg.norm(self.sensor_node_positions[indice] - self.auv_position)
            e = self.energy_required_for_trans(r,beta)
            e_value[indice] = e
            if self.energy_stored[indice] > e:
                E_n[indice] = 1
        return E_n, e_value

    def update_Age(self, node_selected_index):
        self.AoI_all_nodes[node_selected_index] = 1
        for i in range(len(self.AoI_all_nodes)):
            if i != node_selected_index:
                self.AoI_all_nodes[i] = min(self.AoI_max, self.AoI_all_nodes[i] + 1)
        return self.AoI_all_nodes

    def update_all_Age(self):
        for i in range(len(self.AoI_all_nodes)):
            self.AoI_all_nodes[i] = min(self.AoI_max, self.AoI_all_nodes[i] + 1)
        return self.AoI_all_nodes

    def get_cumulative_rewards(self):
        return self.cumulative_rewards

    def find_nearest_node_index(self):
        distances = [np.linalg.norm(sensor_node_position - self.auv_position) for sensor_node_position in self.sensor_node_positions]
        return np.argmin(distances)

    def get_possible_directions(self):
        possible_mvt = np.ones(6)  # because we have directions
        if self.auv_position[0] == 1:
            possible_mvt[1] = 0
        if self.auv_position[0] == 10:
            possible_mvt[0] = 0
        if self.auv_position[1] == 1:
            possible_mvt[3] = 0
        if self.auv_position[1] == 10:
            possible_mvt[2] = 0
        if self.auv_position[2] == 1:
            possible_mvt[5] = 0
        if self.auv_position[2] == 4:
            possible_mvt[4] = 0
        possible_directions = np.where(possible_mvt == 1)[0]
        return possible_directions

    def render(self):
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()
        self.screen.fill((255, 255, 255))

        cell_size = self.window_size // 10
        for i in range(11):
            pygame.draw.line(self.screen, (100, 100, 100), (i * cell_size, 0), (i * cell_size, self.window_size), 1)
            pygame.draw.line(self.screen, (100, 100, 100), (0, i * cell_size), (self.window_size, i * cell_size), 1)

        auv_position = self.auv_position
        auv_x = (auv_position[0] - 1) * cell_size + cell_size // 2
        auv_y = (auv_position[1] - 1) * cell_size + cell_size // 2
        pygame.draw.circle(self.screen, (0, 0, 255), (auv_x, auv_y), cell_size // 4)

        for sensor_node_pos in self.sensor_node_positions:
            node_x = (sensor_node_pos[0] - 1) * cell_size + cell_size // 2
            node_y = (sensor_node_pos[1] - 1) * cell_size + cell_size // 2
            pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(node_x - cell_size // 8, node_y - cell_size // 8, cell_size // 4, cell_size // 4))

        cog_x = int((self.center_of_gravity[0] - 1) * cell_size + cell_size // 2)
        cog_y = int((self.center_of_gravity[1] - 1) * cell_size + cell_size // 2)
        pygame.draw.circle(self.screen, (0, 255, 0), (cog_x, cog_y), cell_size // 4)

        pygame.display.update()

        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.time.delay(100)

    def _render_frame(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
