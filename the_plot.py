import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv(".\PPO_21.csv")
df = df[df["Step"] <= 550000]

plt.plot(df["Step"], df["Value"])
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Learning Curve for 3 nodes")
plt.grid(True)
plt.show()
