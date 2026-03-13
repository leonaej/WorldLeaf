import pandas as pd

seeds_dispersed_by = [
    # Elephant disperses acacia and fig seeds
    ('Loxodonta africana', 'disperses_seeds_of', 'Vachellia tortilis'),
    ('Loxodonta africana', 'disperses_seeds_of', 'Vachellia nilotica'),
    ('Loxodonta africana', 'disperses_seeds_of', 'Vachellia xanthophloea'),
    ('Loxodonta africana', 'disperses_seeds_of', 'Ficus sycomorus'),
    ('Loxodonta africana', 'disperses_seeds_of', 'Ficus glumosa'),
    ('Loxodonta africana', 'disperses_seeds_of', 'Kigelia africana'),

    # Giraffe disperses acacia seeds
    ('Giraffa tippelskirchi', 'disperses_seeds_of', 'Vachellia tortilis'),
    ('Giraffa tippelskirchi', 'disperses_seeds_of', 'Vachellia drepanolobium'),
    ('Giraffa tippelskirchi', 'disperses_seeds_of', 'Vachellia xanthophloea'),
    ('Giraffa tippelskirchi', 'disperses_seeds_of', 'Vachellia seyal'),

    # Baboon disperses fig and acacia seeds
    ('Papio anubis', 'disperses_seeds_of', 'Ficus sycomorus'),
    ('Papio anubis', 'disperses_seeds_of', 'Ficus burkei'),
    ('Papio anubis', 'disperses_seeds_of', 'Vachellia tortilis'),

    # Impala disperses acacia seeds
    ('Aepyceros melampus', 'disperses_seeds_of', 'Vachellia tortilis'),
    ('Aepyceros melampus', 'disperses_seeds_of', 'Vachellia nilotica'),

    # Hornbills disperse fig and sausage tree seeds
    ('Tockus deckeni', 'disperses_seeds_of', 'Ficus sycomorus'),
    ('Tockus deckeni', 'disperses_seeds_of', 'Kigelia africana'),
    ('Bucorvus leadbeateri', 'disperses_seeds_of', 'Ficus sycomorus'),
    ('Bucorvus leadbeateri', 'disperses_seeds_of', 'Kigelia africana'),
    ('Bycanistes brevis', 'disperses_seeds_of', 'Ficus sycomorus'),
    ('Bycanistes brevis', 'disperses_seeds_of', 'Ficus burkei'),
    ('Lophoceros nasutus', 'disperses_seeds_of', 'Ficus glumosa'),

    # Dung beetles disperse seeds through dung
    ('Kheper nigroaeneus', 'disperses_seeds_of', 'Vachellia tortilis'),
    ('Kheper nigroaeneus', 'disperses_seeds_of', 'Vachellia nilotica'),
]

df = pd.DataFrame(seeds_dispersed_by, columns=['subject_label', 'relation', 'object_label'])
df['subject_id'] = ''
df['object_id'] = ''
df.to_csv('data/raw/edges_manual_dispersal.csv', index=False)
print(f'Saved {len(df)} disperses_seeds_of edges')
print(df[['subject_label', 'relation', 'object_label']].to_string())