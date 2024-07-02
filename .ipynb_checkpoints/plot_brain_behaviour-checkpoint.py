
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
import plotly.express as px
import scipy.stats as stats
from create_annotation import *

px.defaults.template = "simple_white"
# import umap_male.csv and umap_female.csv
umap_male = pd.read_csv("umap_male.csv")
umap_female = pd.read_csv("umap_female.csv")

# zscore normalize entropy0, entropy1 and entropy2 from umap_male and umap_female
umap_male['entropy0'] = (umap_male['entropy0'] - umap_male['entropy0'].mean()) / umap_male['entropy0'].std()
umap_male['entropy1'] = (umap_male['entropy1'] - umap_male['entropy1'].mean()) / umap_male['entropy1'].std()
umap_male['entropy2'] = (umap_male['entropy2'] - umap_male['entropy2'].mean()) / umap_male['entropy2'].std()

umap_female['entropy0'] = (umap_female['entropy0'] - umap_female['entropy0'].mean()) / umap_female['entropy0'].std()
umap_female['entropy1'] = (umap_female['entropy1'] - umap_female['entropy1'].mean()) / umap_female['entropy1'].std()
umap_female['entropy2'] = (umap_female['entropy2'] - umap_female['entropy2'].mean()) / umap_female['entropy2'].std()



# DBscan clustering from umap_male with the columns x and y 
X = umap_male[['x','y']]
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

umap_male['labels'] = labels

# save the data
umap_male.to_csv("umap_male.csv", index=False)

#same with umap female
X = umap_female[['x','y']]
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

umap_female['labels'] = labels

#save the data
umap_female.to_csv("umap_female.csv", index=False)

# plot DBSCAN output


"""fig = go.Figure()
fig.add_trace(go.Scatter(
    x=umap_male['x'],
    y=umap_male['y'],
    mode='markers',
    marker=dict(
        color=umap_male['labels'],
        colorscale='Viridis',  # You can choose a different colorscale here
        reversescale=True  # This will make the colors darker
    )
))
fig.update_layout(template='simple_white')
fig.update_layout(title='UMAP behavioral Landscape from DGRP Males')
fig.show()
"""
# 3D scatter plot plotly with entropy0, entropy1 and entropy2

"""

fig = px.scatter(umap_male, x='x', y='y', color='morphology',
                 color_continuous_scale='Viridis', template='simple_white',
                 title='UMAP behavioral Landscape from DGRP Males')

fig.update_traces(marker=dict(size=20,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

##fig.write_html("umap_male.html")
fig.show()


fig = px.scatter_3d(umap_male, x='entropy0', y='entropy1', z='entropy2', color='labels',
                    color_continuous_scale='Viridis', template='simple_white',
                    title='Brain Landscape from DGRP Males')
fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

#fig.write_html("Entropy_space_umap_male.html")
fig.show()


# same with umap_female
 
fig = px.scatter(umap_female, x='x', y='y', color='morphology',
                 color_continuous_scale='Viridis', template='simple_white',
                 title='UMAP behavioral Landscape from DGRP Females')
fig.update_traces(marker=dict(size=20,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

#fig.write_html("umap_female.html")

fig.show()

fig = px.scatter_3d(umap_female, x='entropy0', y='entropy1', z='entropy2', color='labels',
                    color_continuous_scale='Viridis', template='simple_white',
                    title='Brain Landscape from DGRP Females')
fig.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

#fig.write_html("Entropy_space_umap_female.html")
fig.show()

"""

# separate morphology in several list based on their labels




# Create the box plot

# Create the box plot
fig = px.box(umap_female, x='labels', y='morphology', 
            color="labels",
            title='Box Plot of Morphology by behavioral clusters in females'
             )

fig.write_html("boxplot_morphology_behavior_female.html")
fig.show()





# same for males
fig = px.box(umap_male, x='labels', y='morphology',
             color='labels',
             title='Box Plot of Morphology by behavioral clusters in males')

# Extract the data for each group
group1_data = umap_male[umap_male['labels'] == -1]['morphology']
group2_data = umap_male[umap_male['labels'] == 0]['morphology']

# Perform the t-test
t_statistic, p_value = stats.ttest_ind(group1_data, group2_data)

# Add p-value annotation to the plot
fig.add_annotation(
    x=-0.5, y=2.13,
    text=f"P-Value: {p_value:.4f}",
    showarrow=False,
    font=dict(size=14)
)

# Add bbar between the boxplots
fig.add_trace(go.Scatter(
    x=[-1, 0],
    y=[2.12, 2.12],
    mode='lines',
    line=dict(color='black', width=2)
))

fig.write_html("boxplot_morphology_behavior_male.html")
fig.show()
