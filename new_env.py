import gym
from gym import spaces
import numpy as np
import pygame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def signal_to_noise_ratio(SL,TL,NL):

    return SL-TL-NL

def Acoustic_source_level(P_elec,elec_acous_conv_eff,DI):
    
    return 170.8+10*np.log10(P_elec)+10*np.log10(elec_acous_conv_eff)+DI

def Transmission_Loss(f,k,r):

    alfa_THorp = (0.11*(f**2/(f**2+1))
            +44*(f**2/(f**2+4100))+2.75*pow(10,-4)*pow(f,2)+
            0.003)*(10**(-3))
    log_r = np.where(r < 1, 0, np.log10(r))

    return k*log_r+r*alfa_THorp

def Power_harvested(n,RL,RVS,Rp):

    p=10**(RL/20)  #acoustic pressure p on the hydrophone
    #RVS=20*np.log10(M)   Receiving voltage sensitivity (RVS) of a hydrophone
    V_ind=p*(10**(RVS/20)) # induced voltage
    P_available=n*(V_ind**2)/(4*Rp)
    P_har=0.7*P_available
    return P_har
def snr_needed_for_transmission_data(system_throughput,Bandiwdth):

        snr=2**(system_throughput/Bandiwdth)-1

    

        return snr


class AUVEnvironment(gym.Env):
    def __init__(self):
        super(AUVEnvironment, self).__init__()
        self.window_size = 500 #  the window size
        self.render_mode = "human"  # because we need real time visualization the render mode is human
        self.metadata = {"render_fps": 30}  
        self.auv_position = np.array([500.0, 500.0, 200.0]) # position of the auv at the center of network
        self.sensor_node_positions = [
            np.array([100.0, 100.0, 100.0]),
            np.array([800.0, 800.0,200.0]),
            np.array([300.0, 100.0, 300.0]),
            np.array([400.0, 600.0, 400.0]),
            np.array([800.0, 500.0, 300.0]),
            #np.array([6, 2, 2]),
            #np.array([4, 4, 3]),
            #np.array([3, 7, 2]),
            #np.array([5, 8, 4]),
            #np.array([6, 9, 3]),
           

           

          


           

        ]
        self.energy_min=25
        self.num_devices=5
        
        self.center_of_gravity = np.mean(self.sensor_node_positions, axis=0)
        position_low = np.array([0.0, 0.0, 0.0])
        position_high = np.array([1000, 1000, 1000])

        energy_low = np.zeros(self.num_devices)  
        energy_high = np.ones(self.num_devices) * 10000  
        age_low = np.zeros(self.num_devices)  
        age_high = np.ones(self.num_devices) * 50 

        self.AoI_all_nodes=np.ones(self.num_devices)
        self.max_iterations=100

        self.AoI_max=self.max_iterations/2
        self.reward_per_step=[]
        self.action_space = spaces.Box(low=np.array([-100,-100,-100,0]), high=np.array([100,100,100,4.999999]))
        
        self.observation_space = spaces.Box(low=np.concatenate((position_low, energy_low, age_low)), 
                                             high=np.concatenate((position_high, energy_high, age_high)),
                                             dtype=np.float32)  
              
        self.energy_stored = np.ones(self.num_devices)*self.energy_min
        self.energy_harvested=0
        self.t=0
        self.f=0
        self.n=0
        self.occurence=np.zeros(self.num_devices)
    def step(self, action):
        reward=0
        q_x=action[0]
        q_y=action[1]
        q_z=action[2]
        selection_node_wet=int(action[3])
        self.auv_position += np.array([q_x, q_y, q_z])
        for i in range (3):
            if(self.auv_position[i]>1000):
                reward -=5
                self.auv_position[i]=1000
            if(self.auv_position[i]<0):
                reward -=5
                self.auv_position[i]=0

        
        
        selected_sensor_node = self.sensor_node_positions[selection_node_wet]
        #selected_sensor_node_collect_data=self.sensor_node_positions[selection_node_collect_data]
        harvested_energy = self.compute_harvested_energy(selected_sensor_node)
        #reward += harvested_energy
        self.energy_stored[selection_node_wet] += harvested_energy  
        self.energy_harvested +=harvested_energy
        """"
        if (self.nodes_in_coverage_area()==[]):
            reward -=1
            self.AoI_all_nodes=self.update_all_Age()        
        else :
                for i in range(len(self.energy_stored)):
                    if i not in self.nodes_in_coverage_area():
                        self.AoI_all_nodes[i] += 1
                    elif self.energy_stored[i] > 55 + self.energy_min:
                        self.energy_stored[i] -= 55
                        self.AoI_all_nodes[i] = 1
                        self.occurence[i] += 1

                    else:
                         self.AoI_all_nodes[i] += 1"""
        E_n,e_values=self.indices_state_for_transmitting()

        available_indices_for_transmission = [i for i, val in enumerate(E_n) if val == 1]
        if selection_node_wet not in available_indices_for_transmission:
            
            if (len(available_indices_for_transmission)==0):
                """selection_node_data= np.random.choice(available_indices_for_transmission)

                self.energy_stored[selection_node_data] -=e_values[selection_node_data]
                AoI=self.update_Age(selection_node_data)
                reward -=3
                self.f +=1"""
                AoI=self.update_all_Age()
                self.n +=1
                reward -=1


            
            else :
                reward -=8
                AoI=self.update_all_Age()
                

        else:
            
             self.energy_stored[selection_node_wet] -=e_values[selection_node_wet]
             self.occurence[selection_node_wet] +=1
             AoI=self.update_Age(selection_node_wet)
             self.t +=1
             if(self.occurence[selection_node_wet] >18):
                 reward -=10

            
        
        num_nodes_needs_to_send_data = sum(1 for x in AoI if x == 50)  

        sum_of_squares = sum(x**2 for x in self.occurence)
        sum_of_values = sum(self.occurence)

        Jain_index= ((sum_of_values**2)/(sum_of_squares*self.num_devices) )if sum_of_squares != 0 else 0

       # print("jain",Jain_index)

        #print("occ",self.occurence)

       
        num_zeros = sum(1 for x in self.occurence if x == 0)  
        
        reward -=(1-Jain_index)*(np.sum(AoI)/self.num_devices)+num_zeros
        

        #reward -=10*max(0,(num_zeros/(self.num_devices)))
        
        #reward -= 5*max(0,1-(self.t/self.n))

        self.max_iterations -= 1
        state = self._get_observation()
        if  self.max_iterations <=0 :
        
            done = True
        else:
            done = False
        
        return state, reward, done,{}

    def reset(self):

        self.auv_position = np.array([500.0, 500.0, 200.0]) # position of the auv at the center of network
        self.max_iterations=100
        self.AoI_all_nodes=np.ones(self.num_devices)    
        self.energy_stored = np.ones(self.num_devices)*self.energy_min
        self.energy_harvested=0
        self.occurence=np.zeros(self.num_devices)
        self.t=0
        self.f=0
        self.n=0

        return     np.hstack((self.auv_position,self.energy_stored,self.AoI_all_nodes))

   

    
    def _get_observation(self):
        
        return     np.hstack((self.auv_position,self.energy_stored,self.AoI_all_nodes))

    
    
    def compute_harvested_energy(self, sensor_node_position):
        SL=Acoustic_source_level(2000,0.5,20)
        #avg_distance=0.5*(self.auv_position+self.prev_auv_position)
        r = np.linalg.norm(sensor_node_position - self.auv_position)
        #print("auv pos",self.auv_position," sensor",sensor_node_position)
        AL=Transmission_Loss(60,1.5,r)
        NL=30
        RVS=-150
        Rp=125
        duration=25 #25 seconds
        RL=signal_to_noise_ratio(SL,AL,NL)
        P_harvested=Power_harvested(2,RL,RVS,Rp)

        energy_harvested=P_harvested*duration
        return energy_harvested
    

    def energy_required_for_trans(self,sensor_node_position):
        snr=snr_needed_for_transmission_data(4,3000)

        r = np.linalg.norm(sensor_node_position -self.auv_position)
        AL=Transmission_Loss(20,1.5,r)
        NL=30
        power_for_transmission=snr*(10**(AL/10))*(10**(NL/10))
        duration=25
        energy_for_tranmission=power_for_transmission*duration

        return energy_for_tranmission

    def indices_state_for_transmitting(self):
        E_n=[0]*self.num_devices
        e_value=[0]*self.num_devices
        for indice in range (len(self.sensor_node_positions)):
            e=55+self.energy_min
            e_value[indice]=e
            if(self.energy_stored [indice]>e):
                E_n[indice]=1

        return E_n,e_value
        
        
    def update_Age(self,node_selected_index):
        self.AoI_all_nodes[node_selected_index]=1
        for i in range(len(self.AoI_all_nodes)):
            if i != node_selected_index:  
                self.AoI_all_nodes[i] =  min(self.AoI_max,self.AoI_all_nodes[i] + 1)
                #self.AoI_all_nodes[i] =  self.AoI_all_nodes[i] + 1
        return self.AoI_all_nodes
    
    def update_all_Age(self):
        for i in range(len(self.AoI_all_nodes)):
                self.AoI_all_nodes[i] =  min(self.AoI_max,self.AoI_all_nodes[i] + 1)
                #self.AoI_all_nodes[i] =  self.AoI_all_nodes[i] + 1
        return self.AoI_all_nodes
    
    
        

    
    def nodes_in_coverage_area(self):
        
        coverage_indices = []
        auv_pos = self.auv_position
        for i, node_pos in enumerate(self.sensor_node_positions):
            # Check if node's position falls within the coverage area around the AUV
            if (np.abs(node_pos[0] - auv_pos[0]) <= 100 and
                np.abs(node_pos[1] - auv_pos[1]) <= 100 and
                np.abs(node_pos[2] - auv_pos[2]) <= 100):
                coverage_indices.append(i)
        return coverage_indices

    def get_cumulative_rewards(self):
        return self.cumulative_rewards

    def find_nearest_node_index(self):
        distances = [np.linalg.norm(sensor_node_position - self.auv_position) for sensor_node_position in self.sensor_node_positions]
        return np.argmin(distances)
    
   #  the action mask function is used so we won't get outside of the grid boundaries
    
    
    def render(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot AUV
        auv_x, auv_y, auv_z = self.auv_position
        ax.scatter(auv_x, auv_y, auv_z, c='b', marker='o', label='AUV')

        # Plot sensor nodes
        for i, node_pos in enumerate(self.sensor_node_positions):
            node_x, node_y, node_z = node_pos
            ax.scatter(node_x, node_y, node_z, c='r', marker='s', label=f'Node {i}')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('AUV Environment')
        ax.legend()

        plt.show()