"""
Visualize the NEEDLE DAG structure from snapshot.
"""

import matplotlib.pyplot as plt
import networkx as nx

from needle.utils.results import DAGSnapshot

# Load snapshot
snapshot = DAGSnapshot.from_json("runs/dag_snapshot.json")

# Build directed graph
G = nx.DiGraph()

# Add edges with metadata
for edge in snapshot.edges:
    for source in edge.source_nodes:
        G.add_edge(source, edge.target_node, method=edge.method.value, label=edge.method.value)


# Create hierarchical layout manually
def get_node_level(node):
    """Assign level based on node type (higher level = top of tree)"""
    if node == "root":
        return 4  # Top level
    elif "estimator" in node and "systematic" not in node:
        return 3  # Estimators
    elif "systematic" in node and "ensemble" not in node:
        return 2  # Systematics
    elif "ensemble" in node and "fold" not in node:
        return 1  # Ensembles
    elif "fold" in node:
        return 0  # Folds (bottom)
    return 0


# Group nodes by level
levels = {}
for node in G.nodes():
    level = get_node_level(node)
    if level not in levels:
        levels[level] = []
    levels[level].append(node)

# Assign positions
pos = {}
for level, nodes in levels.items():
    # Spread nodes horizontally at each level
    num_nodes = len(nodes)
    for i, node in enumerate(sorted(nodes)):
        x = (i - num_nodes / 2) * 3  # Horizontal spacing
        y = level * 2  # Vertical spacing (higher level = higher y)
        pos[node] = (x, y)

# Create figure
fig, ax = plt.subplots(figsize=(24, 14))

# Color nodes by level
node_colors = []
for node in G.nodes():
    if node == "root":
        node_colors.append("#FF6B6B")  # Red for root
    elif "fold" in node:
        node_colors.append("#964ECD")  # Purple for folds
    elif "ensemble" in node:
        node_colors.append("#95E1D3")  # Light green for ensembles
    elif "systematic" in node:
        node_colors.append("#36C182")  # Green for systematics
    else:
        node_colors.append("#FEE715")  # Yellow for estimators

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, alpha=0.9, ax=ax)

# Draw edges with labels
nx.draw_networkx_edges(
    G,
    pos,
    edge_color="gray",
    arrows=True,
    arrowsize=20,
    width=2,
    connectionstyle="arc3,rad=0.1",  # Curved edges for better visibility
    ax=ax,
)

# Draw labels (shortened for readability)
labels = {}
for node in G.nodes():
    parts = node.split("_")

    if node == "root":
        labels[node] = "ROOT"
    else:
        labels[node] = f"{parts[-2]}_{parts[-1]}"

nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold", ax=ax)

# Draw edge labels (aggregation methods)
edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9, font_color="red", ax=ax)

# Add level labels on the left side
level_names = {4: "Root", 3: "Estimators", 2: "Systematics", 1: "Ensembles", 0: "Folds"}
for level, name in level_names.items():
    if level in [l for n in G.nodes() for l in [get_node_level(n)] if l == level]:
        ax.text(-15, level * 2, name, fontsize=14, fontweight="bold", ha="right", va="center")

# Title and legend
plt.title("NEEDLE DAG Structure (Hierarchical)", fontsize=22, fontweight="bold", pad=20)

# Add legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="#FF6B6B", label="Root"),
    Patch(facecolor="#FEE715", label="Estimators"),
    Patch(facecolor="#36C182", label="Systematics"),
    Patch(facecolor="#95E1D3", label="Ensembles"),
    Patch(facecolor="#964ECD", label="Folds"),
]
plt.legend(handles=legend_elements, loc="upper right", fontsize=12)

plt.axis("off")
plt.tight_layout()
plt.savefig("dag_structure.png", dpi=300, bbox_inches="tight")
plt.savefig("dag_structure.pdf", bbox_inches="tight")  # Vector format
print("DAG visualization saved to dag_structure.png and dag_structure.pdf")
