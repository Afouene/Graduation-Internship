import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv("PPO_55.csv")
df = df[df["Step"] <= 260000]

plt.plot(df["Step"], df["Value"])
plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Learning Curve for 5 nodes")
plt.grid(True)
plt.show()