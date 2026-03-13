import pandas as pd

symbiotic = [
    # Oxpeckers and large mammals (mutualism - oxpecker eats ticks, mammal gets cleaned)
    ('Buphagus africanus', 'symbiotic_with', 'Syncerus caffer'),
    ('Buphagus africanus', 'symbiotic_with', 'Equus quagga'),
    ('Buphagus africanus', 'symbiotic_with', 'Diceros bicornis'),
    ('Buphagus africanus', 'symbiotic_with', 'Giraffa tippelskirchi'),
    ('Buphagus africanus', 'symbiotic_with', 'Connochaetes taurinus'),
    ('Buphagus africanus', 'symbiotic_with', 'Phacochoerus africanus'),
    ('Buphagus erythroryncha', 'symbiotic_with', 'Syncerus caffer'),
    ('Buphagus erythroryncha', 'symbiotic_with', 'Equus quagga'),
    ('Buphagus erythroryncha', 'symbiotic_with', 'Diceros bicornis'),
    ('Buphagus erythroryncha', 'symbiotic_with', 'Giraffa tippelskirchi'),
    ('Buphagus erythroryncha', 'symbiotic_with', 'Connochaetes taurinus'),

    # Cattle egret and buffalo (egret eats insects disturbed by buffalo)
    ('Ardea ibis', 'symbiotic_with', 'Syncerus caffer'),
    ('Ardea ibis', 'symbiotic_with', 'Loxodonta africana'),
    ('Ardea ibis', 'symbiotic_with', 'Connochaetes taurinus'),
    ('Ardea ibis', 'symbiotic_with', 'Equus quagga'),

    # Mongoose and warthog (share burrows)
    ('Mungos mungo', 'symbiotic_with', 'Phacochoerus africanus'),
    ('Helogale parvula', 'symbiotic_with', 'Phacochoerus africanus'),

    # Zebra and wildebeest (mutual warning system)
    ('Equus quagga', 'symbiotic_with', 'Connochaetes taurinus'),
    ('Connochaetes taurinus', 'symbiotic_with', 'Equus quagga'),

    # Baboon and elephant (baboons alert elephants to predators)
    ('Papio anubis', 'symbiotic_with', 'Loxodonta africana'),
    ('Loxodonta africana', 'symbiotic_with', 'Papio anubis'),

]

df = pd.DataFrame(symbiotic, columns=['subject_label', 'relation', 'object_label'])
df['subject_id'] = ''
df['object_id'] = ''
df.to_csv('data/raw/edges_manual_symbiotic.csv', index=False)
print(f'Saved {len(df)} symbiotic_with edges')
print(df[['subject_label', 'relation', 'object_label']].to_string())