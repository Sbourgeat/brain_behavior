import pandas as pd
import plotly.express as px

# Import the KL divergence
kl = pd.read_csv('kl_divergence_normal.csv', header=None)

# Set the first row as the column names
kl.columns = kl.iloc[0, :]
kl = kl.iloc[1:, :]

# Copy the genotypes from the first row to the first column
kl.iloc[:, 0] = kl.columns

# Set the first column as the index
kl.set_index(kl.columns[0], inplace=True)
kl.index.name = None

print(kl)
# Plot the KL divergence as a heatmap
fig = px.imshow(kl)

# Add legends and axis names
fig.update_layout(
    title="KL Divergence Heatmap",
    xaxis_title="Genotype",
    yaxis_title="Genotype",
    coloraxis_colorbar=dict(
        title="KL Divergence",
    )
)

fig.show()