import pandas as pd

species = pd.read_csv('data/raw/inaturalist_species_filtered.csv')
species_names = set(species['name'].tolist())
symbiotic = pd.read_csv('data/raw/edges_manual_symbiotic.csv')

print('SYMBIOTIC - checking both sides:')
for _, row in symbiotic.iterrows():
    subj_ok = row['subject_label'] in species_names
    obj_ok = row['object_label'] in species_names
    if not subj_ok or not obj_ok:
        print(f"  MISSING: {row['subject_label']} ({subj_ok}) -> {row['object_label']} ({obj_ok})")
print('Done!')