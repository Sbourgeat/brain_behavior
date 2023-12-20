"""
Here, we will compare the KL divergence between each genotypes scored with 
the bottleneck distances of the same genotypes.
"""

import pandas as pd
import plotly.express as px

# import the KL divergence
kl = pd.read_csv('kl_divergence.csv')

# import the bottleneck distances
bottleneck = pd.read_csv('bottleneck.csv')


# filter both dataframes to keep only the rows sharing the same DGRP_lines
kl = kl[kl['DGRP_line'].isin(bottleneck['DGRP_line'])]
bottleneck = bottleneck[bottleneck['DGRP_line'].isin(kl['DGRP_line'])]

# compare the kl divergence and the bottleneck distances by plotting  a scatter plot plotly
fig = px.scatter(kl, x="kl_divergence", y="bottleneck", hover_data=['DGRP_line'])
fig.show()

# trace the trendline and add the r2 value and the pvalue to the plot
fig = px.scatter(kl, x="kl_divergence", y="bottleneck", hover_data=['DGRP_line'], trendline="ols")
# calculate and add R2 to the plot
fig.add_annotation(
    x=0.5,
    y=0.9,
    text="R2 = " + str(round(fig.data[0].meta['r_squared'], 3)),
    showarrow=False,
    font=dict(
        family="Courier New, monospace",
        size=16,
        color="#ffffff"
    ),
    align="center",
    bordercolor="#c7c7c7",
    borderwidth=2,
    borderpad=4,
    bgcolor="#ff7f0e",
    opacity=0.8
)

# calculate and add the pvalue for pearson correlation to the plot
fig.add_annotation(
    x=0.5,
    y=0.8,
    text="p-value = " + str(round(fig.data[0].meta['p_value'], 3)),
    showarrow=False,
    font=dict(
        family="Courier New, monospace",
        size=16,
        color="#ffffff"
    ),
    align="center",
    bordercolor="#c7c7c7",
    borderwidth=2,
    borderpad=4,
    bgcolor="#ff7f0e",
    opacity=0.8
)
fig.update_layout(
    title="KL divergence vs Bottleneck distance",
    xaxis_title="KL divergence",
    yaxis_title="Bottleneck distance",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)
fig.show()

