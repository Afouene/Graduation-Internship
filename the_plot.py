import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv(".\PPO_20.csv")
df = df[df["Step"] <= 955000]

plt.plot(df["Step"], df["Value"])
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Learning Curve for 7 nodes")
plt.grid(True)
plt.show()
