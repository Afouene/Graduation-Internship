import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv(".\PPO_5.csv")
df = df[df["Step"] <= 135000]

plt.plot(df["Step"], df["Value"])
plt.xlabel("Step")
plt.ylabel("Value")
plt.title("Learning Curve")
plt.grid(True)
plt.show()
