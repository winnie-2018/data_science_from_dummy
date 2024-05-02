# import libraries
import pandas as pd
import matplotlib.pyplot as plt

# read CSV from datasets folder
df = pd.read_csv("datasets/carparkentry.csv")

# Plot graph
fig, ax = plt.subplots()
ax.plot(df['dom'], df['ent'])

# Save this PNG in figures folder
fig.savefig("figures/dom vs ent.png")