# import libraries
import matplotlib.pyplot as plt
import pandas as pd

# read CSV from datasets folder
df = pd.read_csv("datasets/carparkentry.csv")

# Plot graph by setting x and y axis
df.plot(x="dom", y="ent")

# # Save this PNG in figures folder
plt.savefig("figures/entryvsdayofmonth.png") 