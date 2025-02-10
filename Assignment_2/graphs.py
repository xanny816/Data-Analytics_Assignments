import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# Load data
bar_data = pd.read_csv('bar_assignment.csv')
sankey_data = pd.read_csv('sankey_assignment.csv')

# Transform 1 into “Yes” and 0 into “No” for bar chart
bar_data['COUNT'] = bar_data['COUNT'].map({1: 'Yes', 0: 'No'})
bar_grouped = bar_data.groupby(['LABEL', 'COUNT']).size().unstack(fill_value=0)

# Create horizontal stacked bar chart
plt.figure(figsize=(10, 6))
bar_grouped.plot(kind='barh', stacked=True, color=['lightblue', 'orange'])
plt.xlabel('Count')
plt.ylabel('Label')
plt.title('Horizontal Stacked Bar Chart')
plt.legend(title='Count')
plt.savefig('bar_graph.png', bbox_inches='tight')
plt.close()

# Create Sankey diagram
labels = list(sankey_data['LABEL']) + ['Reg', 'Aca', 'Oth']
source = []
target = []
value = []

color_mapping = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3"]  # Node colors
link_colors = []  # Link colors corresponding to node colors

for i, row in sankey_data.iterrows():
    for j, col in enumerate(sankey_data.columns[1:]):
        if row[col] > 0:
            source.append(j)
            target.append(len(sankey_data) + i)
            value.append(row[col])
            link_colors.append(color_mapping[j % len(color_mapping)])  # Use modulo to avoid index out of range

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=20,  # Increased padding for clearer separation
        thickness=30,  # Increased thickness for better visibility
        line=dict(color="black", width=0.7),
        label=labels,
        color=color_mapping
    ),
    link=dict(
        source=source,
        target=target,
        value=value,
        color=link_colors  # Color links based on their corresponding nodes
    ))])
fig.update_layout(
    title_text="Sankey Diagram: Source → Intermediate → Target",
    font_size=12,
    plot_bgcolor='white'  # Clean white background
)
fig.write_image("sankey_diagram.png")

# Create network graph
G = nx.Graph()
central_nodes = ['D', 'F', 'I', 'N', 'S']
other_nodes = ['BIH', 'GEO', 'ISR', 'MNE', 'SRB', 'CHE', 'TUR', 'UKR', 'GBR', 'AUS', 'HKG', 'USA',
               'AUT', 'BEL', 'BGR', 'HRV', 'CZE', 'EST', 'FRA', 'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LUX', 'NLD', 'PRT', 'ROU', 'SVK', 'SVN', 'ESP']

G.add_nodes_from(central_nodes, color='blue')
G.add_nodes_from(other_nodes[:12], color='green')
G.add_nodes_from(other_nodes[12:], color='yellow')

# Create edges
for node in central_nodes:
    G.add_edges_from([(node, other) for other in other_nodes])

# Define custom layout with central nodes in pentagon formation and other nodes in circular layout
pos = {}

# Position central nodes in a pentagon
import math
num_central_nodes = len(central_nodes)
radius = 1.5
for i, node in enumerate(central_nodes):
    angle = 2 * math.pi * i / num_central_nodes
    pos[node] = (radius * math.cos(angle), radius * math.sin(angle))

# Position other nodes in a circle around the central nodes
num_other_nodes = len(other_nodes)
radius_outer = 3
for i, node in enumerate(other_nodes):
    angle = 2 * math.pi * i / num_other_nodes
    pos[node] = (radius_outer * math.cos(angle), radius_outer * math.sin(angle))

# Colors for nodes
colors = [G.nodes[node]['color'] for node in G.nodes]

plt.figure(figsize=(10, 10))
nx.draw(G, pos, node_color=colors, with_labels=True, node_size=2000, font_size=10)
plt.title("Network Graph")
plt.savefig("network_graph.png", bbox_inches='tight')
plt.close()

# Collate graphs into one image with the specified layout
bar_img = Image.open('bar_graph.png')
sankey_img = Image.open('sankey_diagram.png')
network_img = Image.open('network_graph.png')

# Create a figure with a custom grid layout
fig = plt.figure(figsize=(20, 12))  # Adjust figure size as needed

# Define grid layout: 2 rows, 2 columns
# Left column: Bar Chart (top) and Sankey Diagram (bottom)
# Right column: Network Plot (spanning both rows)
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5])  # Adjust width ratios as needed

# Bar Chart (top-left)
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(bar_img)
ax1.axis('off')
ax1.set_title('Bar Chart', pad=20)

# Sankey Diagram (bottom-left)
ax2 = fig.add_subplot(gs[1, 0])
ax2.imshow(sankey_img)
ax2.axis('off')
ax2.set_title('Sankey Diagram', pad=20)

# Network Plot (right side, spanning both rows)
ax3 = fig.add_subplot(gs[:, 1])  # Span all rows in the second column
ax3.imshow(network_img)
ax3.axis('off')
ax3.set_title('Network Plot', pad=20)

# Adjust layout to prevent overlap
plt.tight_layout(pad=3.0)
plt.savefig('collated_graphs.png', bbox_inches='tight')
plt.close()

print("Graphs generated and saved successfully.")
