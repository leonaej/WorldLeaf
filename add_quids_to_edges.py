import pandas as pd

species = pd.read_csv('data/raw/species_with_qids_filtered.csv')
name_to_qid = dict(zip(species['name'].str.strip(), species['wikidata_qid']))

edges = pd.read_csv('data/raw/edges_all_filtered.csv')

def fill_qid(label, existing_id):
    if pd.notna(existing_id) and str(existing_id).strip() != '':
        return existing_id
    return name_to_qid.get(str(label).strip(), '')

edges['subject_id'] = edges.apply(lambda r: fill_qid(r['subject_label'], r['subject_id']), axis=1)
edges['object_id'] = edges.apply(lambda r: fill_qid(r['object_label'], r['object_id']), axis=1)

missing = edges[
    edges['subject_id'].isna() | (edges['subject_id'] == '') |
    edges['object_id'].isna() | (edges['object_id'] == '')
]

print(f'Still missing: {len(missing)}')
edges.to_csv('data/raw/edges_all_filtered.csv', index=False)
print('Saved!')