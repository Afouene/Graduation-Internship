import numpy as np
import matplotlib.pyplot as plt

def snr_needed_for_transmission_data(system_throughput, Bandiwdth):
    snr = 2**(system_throughput / Bandiwdth) - 1
    return snr

def Transmission_Loss(f, k, r):
    alfa_THorp = (0.11 * (f**2 / (f**2 + 1)) + 44 * (f**2 / (f**2 + 4100)) + 2.75 * 10**-4 * f**2 + 0.003) * 10**-3
    log_r = np.where(r < 1, 0, np.log10(r))
    return k * log_r + r * alfa_THorp

def energy_required_for_trans(auv, sensor_node_position, beta):
    snr = snr_needed_for_transmission_data(100 / (25 * (1 - beta)), 3000)
    r = np.linalg.norm(sensor_node_position - auv)
    AL = Transmission_Loss(30, 1.5, 100 * r)
    NL = 30
    power_for_transmission = snr * 10**(AL / 10) * 10**(NL / 10)
    duration =25 * (1-beta)
    energy_for_transmission = power_for_transmission * duration
    return energy_for_transmission

def signal_to_noise_ratio(SL, TL, NL):
    return SL - TL - NL

def Acoustic_source_level(P_elec, elec_acous_conv_eff, DI):
    return 170.8 + 10 * np.log10(P_elec) + 10 * np.log10(elec_acous_conv_eff) + DI

def Power_harvested(n, RL, RVS, Rp):
    p = 10**(RL / 20)  # acoustic pressure p on the hydrophone
    V_ind = p * 10**(RVS / 20)  # induced voltage
    P_available = n * (V_ind**2) / (4 * Rp)
    P_har = 0.7 * P_available
    return P_har

def compute_harvested_energy(auv, sensor_node_position, beta):
    SL = Acoustic_source_level(2000, 0.5, 20)
    r = np.linalg.norm(sensor_node_position - auv)
    AL = Transmission_Loss(40, 1.5, 100 * r)
    NL = 30
    RVS = -150
    Rp = 125
    duration = 25 * beta  # 25 seconds
    RL = signal_to_noise_ratio(SL, AL, NL)
    P_harvested = Power_harvested(2, RL, RVS, Rp)
    energy_harvested = P_harvested * duration
    return energy_harvested

auv = np.array([1, 1, 1])
sensor_node_position = [[4, 5, 5], [5, 4, 5], [5, 5, 4], [4, 4, 5], [5, 4, 4], [4, 5, 4], [4, 4, 4]]

for x in range(1, 11):
    for y in range(1, 11):
        for z in range(1, 5):
            sensor_node_position.append([x, y, z])

def compute_energies(beta):
    distances = []
    transmission_energy = []
    harvested_energy = []
    for position in sensor_node_position:
        distance = np.linalg.norm(position - auv)
        energy = compute_harvested_energy(auv, position, 1)
        energy1 = energy_required_for_trans(auv, position, 0)
        distances.append(distance)
        harvested_energy.append(energy)
        transmission_energy.append(energy1)
    return distances, harvested_energy, transmission_energy

# Compute energies for different beta values
betas = [0.1, 0.5, 0.9]
colors = ['g', 'b', 'r']
labels = ['beta = 0.1', 'beta = 0.5', 'beta = 0.9']

plt.figure(figsize=(8, 6))

for beta, color, label in zip(betas, colors, labels):
    distances, harvested_energy, transmission_energy = compute_energies(beta)
    sorted_indices = np.argsort(distances)
    sorted_distances = np.array(distances)[sorted_indices]
    sorted_energy1 = np.array(transmission_energy)[sorted_indices]
    sorted_energy = np.array(harvested_energy)[sorted_indices]
    plt.plot(100 * sorted_distances, sorted_energy, color=color, linestyle='-', label=f'Energy harvested ({label})')

    plt.plot(100 * sorted_distances, sorted_energy1, color=color, linestyle='--', label=f'Energy needed for transmission ({label})')

plt.xlabel('Distance between auv and a sensor node (meters)')
plt.ylabel('Energy (Joules)')
plt.legend()
plt.grid(True)
plt.show()
