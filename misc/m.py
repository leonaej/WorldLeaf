import pandas as pd

nodes = pd.read_csv(r'C:\leo\github Repository\WorldLeaf\data\processed\nodes.csv')
temp3 = pd.read_csv(r'C:\leo\github Repository\WorldLeaf\data\raw\taxonomy_nodes_temp_3.csv')

merged = temp3.merge(nodes[['node_id', 'name']], on='name', how='left')

merged.to_csv(r'C:\leo\github Repository\WorldLeaf\data\raw\taxonomy_nodes_temp_3.csv', index=False)

print(f"Total rows: {len(merged)}")
print(f"Matched: {merged['node_id'].notna().sum()}")
print(f"Unmatched: {merged['node_id'].isna().sum()}")