import numpy as np
import pandas as pd


def download_file(url, dest_path):
    import requests
    resp = requests.get(url)
    with open(dest_path, 'wb') as f:
        f.write(resp.content)

def parse_classifications(text):
    m = dict()
    parts = text.split('-')
    m['location'] = parts[:-1]
    m['membrane_or_soluable'] = parts[-1]
    return m

def parse_description(desc):
    # remove leading '>'
    desc = desc[1:]

    parts = desc.split()

    m = dict()

    m['id'] = parts[0]
    m.update(parse_classifications(parts[1]))

    is_test = False
    if (len(parts) == 3):
        is_test = parts[2] == 'test'
    m['is_test'] = is_test
    return m

def parse_record(description, sequence):
    description = description.strip()
    sequence = sequence.strip()

    m = parse_description(description)
    m['sequence'] = list(sequence)

    return m

def parse_deeploc_fasta(file):
    records = []
    with open(file, 'r') as f:
        line = f.readline()
        while line:
            desc = line
            seq = f.readline()
            records.append(parse_record(desc, seq))
            line = f.readline()
    return records

def is_test_pred(m):
    return m['is_test']

def is_unknown_pred(m):
    return m['soluable'] == 'U'

def single_loc_pred(m):
    return len(m['location']) == 1

def valid_record_pred(m):
    return single_loc_pred(m)

def training_pred(m):
    return not is_test_pred(m) and not is_unknown_pred(m) and single_loc_pred(m)

def select_valid_records(data):
    return list(filter(valid_record_pred, data))

def limit_sequence(xs):
    if len(xs) > 1000:
        return xs[:500] + xs[-500:]
    else:
        return xs

def encode_blosum62(df):
    blosum62_matrix = {
        'A' : [  4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0 ],
        'R' : [ -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3 ],
        'N' : [ -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3 ],
        'D' : [ -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3 ],
        'C' : [  0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1 ],
        'Q' : [ -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2 ],
        'E' : [ -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2 ],
        'G' : [  0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3 ],
        'H' : [ -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3 ],
        'I' : [ -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3 ],
        'L' : [ -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1 ],
        'K' : [ -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2 ],
        'M' : [ -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1 ],
        'F' : [ -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1 ],
        'P' : [ -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2 ],
        'S' : [  1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2 ],
        'T' : [  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0 ],
        'W' : [ -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3 ],
        'Y' : [ -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1 ],
        'V' : [  0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4 ]}
    total = df['id'].count()
    data = np.zeros((total, 1000, 20))
    i = 0
    for r, row in df.iterrows():
        for j, a in enumerate(row['sequence_limited']):
            if a in blosum62_matrix:
                data[i][j][:20] = blosum62_matrix[a]
        i = i+1
    return data

def encode_blomap(df):
    blomap = {
        'A': [-0.57,  0.39, -0.96, -0.61, -0.69],
        'R': [-0.40, -0.83, -0.61,  1.26, -0.28],
        'N': [-0.70, -0.63, -1.47,  1.02,  1.06],
        'D': [-1.62, -0.52, -0.67,  1.02,  1.47],
        'C': [ 0.07,  2.04,  0.65, -1.13, -0.39],
        'Q': [-0.05, -1.50, -0.67, -0.49,  0.21],
        'E': [-0.64, -1.59, -0.39,  0.69,  1.04],
        'G': [-0.90,  0.87, -0.36,  1.08,  1.95],
        'H': [ 0.73, -0.67, -0.42,  1.13,  0.99],
        'I': [ 0.59,  0.79,  1.44, -1.90, -0.93],
        'L': [ 0.65,  0.84,  1.25, -0.99, -1.90],
        'K': [-0.64, -1.19, -0.65,  0.68, -0.13],
        'M': [ 0.76,  0.05,  0.06, -0.62, -1.59],
        'F': [ 1.87,  1.04,  1.28, -0.61, -0.16],
        'P': [-1.82, -0.63,  0.32,  0.03,  0.68],
        'S': [-0.39, -0.27, -1.51, -0.25,  0.31],
        'T': [-0.04, -0.30, -0.82, -1.02, -0.04],
        'W': [ 1.38,  1.69,  1.91,  1.07, -0.05],
        'Y': [ 1.75,  0.11,  0.65,  0.21, -0.41],
        'V': [-0.02,  0.30,  0.97, -1.55, -1.16]}
    total = df['id'].count()
    data = np.zeros((total, 1000, 20))
    i = 0
    for r, row in df.iterrows():
        for j, a in enumerate(row['sequence_limited']):
            if a in blomap:
                data[i][j][:5] = blomap[a]
        i = i+1
    return data

def encode_onehot(df):
    total = df['id'].count()
    data = np.zeros((total, 1000, 20))
    i = 0
    for r, row in df.iterrows():
        for j, a in enumerate(row['sequence_limited']):
            if (a == 'A'):
                data[i][j][0] = 1
            elif (a == 'R'):
                data[i][j][1] = 1
            elif (a == 'N'):
                data[i][j][2] = 1
            elif (a == 'D'):
                data[i][j][3] = 1
            elif (a == 'C'):
                data[i][j][4] = 1
            elif (a == 'Q'):
                data[i][j][5] = 1
            elif (a == 'E'):
                data[i][j][6] = 1
            elif (a == 'G'):
                data[i][j][7] = 1
            elif (a == 'H'):
                data[i][j][8] = 1
            elif (a == 'I'):
                data[i][j][9] = 1
            elif (a == 'L'):
                data[i][j][10] = 1
            elif (a == 'K'):
                data[i][j][11] = 1
            elif (a == 'M'):
                data[i][j][12] = 1
            elif (a == 'F'):
                data[i][j][13] = 1
            elif (a == 'P'):
                data[i][j][14] = 1
            elif (a == 'S'):
                data[i][j][15] = 1
            elif (a == 'T'):
                data[i][j][16] = 1
            elif (a == 'W'):
                data[i][j][17] = 1
            elif (a == 'Y'):
                data[i][j][18] = 1
            elif (a == 'V'):
                data[i][j][19] = 1
            elif (a == 'B'):
                pass
            elif (a == 'U'):
                pass
            elif (a == 'X'):
                pass
            elif (a == 'Z'):
                pass
        i = i+1
    return data

def encode_label(df):
    total = df['id'].count()
    data = np.empty((total))
    i = 0
    for r, row in df.iterrows():
        l = row['location']
        if (l == 'Nucleus'):
            data[i] = 0
        elif (l == 'Cytoplasm'):
            data[i] = 1
        elif (l == 'Extracellular'):
            data[i] = 2
        elif (l == 'Mitochondrion'):
            data[i] = 3
        elif (l == 'Cell.membrane'):
            data[i] = 4
        elif (l == 'Endoplasmic.reticulum'):
            data[i] = 5
        elif (l == 'Plastid'):
            data[i] = 6
        elif (l == 'Golgi.apparatus'):
            data[i] = 7
        elif (l == 'Lysosome/Vacuole'):
            data[i] = 8
        elif (l == 'Peroxisome'):
            data[i] = 9
        i = i+1

    return data

def encode_mask(df):
    total = df['id'].count()
    data = np.zeros((total, 1000))
    i = 0
    for r, row in df.iterrows():
        for j, a in enumerate(row['sequence_limited']):
            data[i][j] = 1
        i = i+1
    return data

def encode_partition(df):
    total = df['id'].count()
    data = np.zeros((total))
    for i in range(total):
        data[i] = (i % 4) + 1
    return data

def rebalance(location):
    import random
    p = {
        'Nucleus': (1271/3235),
        'Cytoplasm': (800/ 2034),
        'Extracellular': (621 / 1580),
        'Mitochondrion': (475 / 1208),
        'Cell.membrane': (419 / 1067),
        'Endoplasmic.reticulum': 271 / 689,
        'Plastid': 234 / 605,
        'Golgi.apparatus': 1.0,
        'Lysosome/Vacuole': 1.0,
        'Peroxisome': 1.0}
    return random.random() < p[location]

#rebalance_index = df_train['location'].apply(rebalance)
#df_train_balanced = df_train.loc[rebalance_index == True]


#
# Local deeploc fasta data file
#
data_file = 'data/deeploc_data.fasta'

#
# Remote deeploc fasta data file
#
data_url = 'http://www.cbs.dtu.dk/services/DeepLoc-1.0/deeploc_data.fasta'


# download remote deeploc data file to local
download_file(data_url, data_file)

# parse deeploc data file
data_all = parse_deeploc_fasta(data_file)

# filter out invalid records (e.g. less record with more than 1 locations)
data_valid = select_valid_records(data_all)

# Load into data frame
df = pd.DataFrame(data_valid)

# convert list to single element
df['location'] = df['location'].apply(lambda x : x[0])

# Need to limit the sequence to 1000 and calculate the length for mask
df['sequence_limited'] =  df['sequence'].apply(limit_sequence)
df['sequence_limited_length'] = df['sequence_limited'].apply(len)
df['sequence_length'] = df['sequence'].apply(len)

# Split into Test and Training Set
df_test = df.loc[df['is_test'] == True]
df_train = df.loc[df['is_test'] == False]

# rebalance training set by boosting rare classes
rebalance_index = df_train['location'].apply(rebalance)
df_train_balanced = df_train.loc[rebalance_index == True]
# df_train_balanced[['id', 'location']].groupby(['location']).count()

def do_encode_blomap():
    np.savez('data/deeploc_balanced_blomap.npz',
             X_train=encode_blomap(df_train_balanced),
             X_test=encode_blomap(df_test),
             mask_train=encode_mask(df_train),
             mask_test=encode_mask(df_test),
             y_train= encode_blomap(df_train_balanced),
             y_test=encode_label(df_test),
             partition= encode_partition(df_train_balanced))

def do_encode_blosum62():
    np.savez('data/deeploc_balanced_blosum62.npz',
             X_train=encode_blosum62(df_train_balanced),
             X_test=encode_blosum62(df_test),
             mask_train=encode_mask(df_train),
             mask_test=encode_mask(df_test),
             y_train=encode_label(df_train_balanced),
             y_test=encode_label(df_test),
             partition=encode_partition(df_train_balanced))

def do_encode_onehot():
    np.savez('data/deeploc_balanced_onehot.npz',
             X_train=encode_onehot(df_train_balanced),
             X_test=encode_onehot(df_test),
             mask_train=encode_mask(df_train),
             mask_test=encode_mask(df_test),
             y_train= encode_label(df_train_balanced),
             y_test=encode_label(df_test),
             partition= encode_partition(df_train_balanced))