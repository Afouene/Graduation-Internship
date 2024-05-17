import numpy as np

def snr_needed_for_transmission_data(system_throughput,Bandiwdth):

        snr=2**(system_throughput/Bandiwdth)-1

    

        return snr

def Transmission_Loss(f,k,r):

    alfa_THorp = (0.11*(f**2/(f**2+1))
            +44*(f**2/(f**2+4100))+2.75*pow(10,-4)*pow(f,2)+
            0.003)*(10**(-3))
    log_r = np.where(r < 1, 0, np.log10(r))

    return k*log_r+r*alfa_THorp

def energy_required_for_trans(auv,sensor_node_position):
        snr=snr_needed_for_transmission_data(4,3000)

        r = np.linalg.norm(sensor_node_position -auv)
        AL=Transmission_Loss(20,1.5,100*r)
        NL=30
        power_for_transmission=snr*(10**(AL/10))*(10**(NL/10))
        duration=25
        energy_for_tranmission=power_for_transmission*duration

        return energy_for_tranmission

auv=np.array([5,5,5])
sensor_node_position=[[4,5,5],[5,4,5],[5,5,4],[4,4,5],[5,4,4],[4,5,4],[4,4,4]]

"""for i in sensor_node_position :
       print(i,"this is " ,energy_required_for_trans(auv,i))"""


def signal_to_noise_ratio(SL,TL,NL):

    return SL-TL-NL

def Acoustic_source_level(P_elec,elec_acous_conv_eff,DI):
    
    return 170.8+10*np.log10(P_elec)+10*np.log10(elec_acous_conv_eff)+DI


def Power_harvested(n,RL,RVS,Rp):

    p=10**(RL/20)  #acoustic pressure p on the hydrophone
    #RVS=20*np.log10(M)   Receiving voltage sensitivity (RVS) of a hydrophone
    V_ind=p*(10**(RVS/20)) # induced voltage
    P_available=n*(V_ind**2)/(4*Rp)
    P_har=0.7*P_available
    return P_har

def compute_harvested_energy(auv, sensor_node_position):
        SL=Acoustic_source_level(2000,0.5,20)
        #avg_distance=0.5*(self.auv_position+self.prev_auv_position)
        r = np.linalg.norm(sensor_node_position - auv)
        #print("auv pos",self.auv_position," sensor",sensor_node_position)
        AL=Transmission_Loss(60,1.5,100*r)
        NL=30
        RVS=-150
        Rp=125
        duration=25 #25 seconds
        RL=signal_to_noise_ratio(SL,AL,NL)
        P_harvested=Power_harvested(2,RL,RVS,Rp)

        energy_harvested=P_harvested*duration
        return energy_harvested




def calcul_c1prime(eta,n,SL,NL,RVS,Rp):
      x=(eta*n)/(4*Rp)
      y=(RVS+SL-2*NL)/10
      
      return x*(10**(y))

def calcul_c2(snr,NL):
      return snr*(10**(NL/10))

def calcul_c3(f):
      alfa_THorp = (0.11*(f**2/(f**2+1))
            +44*(f**2/(f**2+4100))+2.75*pow(10,-4)*pow(f,2)+
            0.003)*(10**(-3))
      return alfa_THorp

SL=Acoustic_source_level(2000,0.5,20)
NL=30
RVS=-150
Rp=125
f=20
c1=calcul_c1prime(0.7,2,SL,NL,RVS,Rp)
snr=snr_needed_for_transmission_data(4,1000)

c2=calcul_c2(snr,NL)
c3=calcul_c3(f)
print("c1=  ",c1,"  c2 =",c2,"  c3 =",c3)