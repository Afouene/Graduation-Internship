import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv(".\PPO_25.csv")
df = df[df["Step"] <= 10000000]
gf = pd.read_csv(".\PPO_30.csv")
gf = gf[gf["Step"] <= 9750000]

plt.plot(df["Step"]/(10**6), df["Value"], label='3D Action Space')
plt.plot(gf["Step"]/(10**6), gf["Value"], label='2D Action Space')

plt.xlabel("Total Time Steps per million")
plt.ylabel("Reward")
plt.legend()
plt.title("Learning Curve for 10 nodes")
plt.grid(True)
plt.show()
