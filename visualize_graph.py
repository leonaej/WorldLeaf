import pickle
import networkx as nx
from pyvis.network import Network

with open('data/processed/graph.gpickle', 'rb') as f:
    G = pickle.load(f)

# Color by relation type
relation_colors = {
    'preys_on':          '#FF4444',
    'eats':              '#FF8800',
    'parent_taxon':      '#4444FF',
    'scavenges_from':    '#AA00AA',
    'migrates_with':     '#00AAAA',
    'disperses_seeds_of':'#00AA00',
    'symbiotic_with':    '#FFFF00',
    'parasitizes':       '#888888',
    'pollinates':        '#FF99CC'
}

# Color nodes by iconic taxon
taxon_colors = {
    'Mammalia':  '#FF6B6B',
    'Aves':      '#4ECDC4',
    'Reptilia':  '#45B7D1',
    'Insecta':   '#96CEB4',
    'Plantae':   '#88D8A3',
    'Amphibia':  '#FFEAA7',
    'unknown':   '#DDD'
}

# Use full graph
subgraph = G
print(f'Visualizing full graph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges')

# Build pyvis network
net = Network(height='900px', width='100%', bgcolor='#1a1a2e', font_color='white', directed=True)
net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=150)

for node, attrs in subgraph.nodes(data=True):
    name = attrs.get('common_name', '') or attrs.get('name', node)
    taxon = attrs.get('iconic_taxon', 'unknown')
    color = taxon_colors.get(taxon, '#DDD')
    size = 15
    net.add_node(node, label=name, color=color, size=size, title=f"{name}\n{taxon}")

for u, v, attrs in subgraph.edges(data=True):
    rel = attrs.get('relation', '')
    color = relation_colors.get(rel, '#999')
    net.add_edge(u, v, color=color, title=rel, arrows='to')

net.save_graph('data/processed/graph_viz.html')
print('Saved → data/processed/graph_viz.html')