import numpy as np
import pygame
from scipy.optimize import bisect

c1 = 0.03366340416928765
c3 = 0.004133836796896648

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
def compute_harvested_energy(r):
        SL=Acoustic_source_level(2000,0.5,20)
        #avg_distance=0.5*(self.auv_position+self.prev_auv_position)
        #print("auv pos",self.auv_position," sensor",sensor_node_position)
        AL=Transmission_Loss(40,1.5,100*r)
        NL=30
        RVS=-150
        Rp=125
        duration=25 #25 seconds
        RL=signal_to_noise_ratio(SL,AL,NL)
        P_harvested=Power_harvested(2,RL,RVS,Rp)

        energy_harvested=P_harvested*duration
        return energy_harvested

def energy_required_for_trans(r):
    snr = snr_needed_for_transmission_data(100 / (25 ), 3000)
    AL = Transmission_Loss(30, 1.5, 100 * r)
    NL = 30
    power_for_transmission = snr * 10**(AL / 10) * 10**(NL / 10)
    duration = 25 
    energy_for_transmission = power_for_transmission * duration
    return energy_for_transmission

print("energy harvested",compute_harvested_energy(r=5))
print("energy required",energy_required_for_trans(r=5))