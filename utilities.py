import os
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import json
import networkx as nx

def safe_mkdir(dirname):
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass

def generate_corpus(df0, cfile, no_kugiri, only_original, validation, code):
    if code!=0:
        df = df0.iloc[:np.max(df0[df0['code']==code].index)+1]
    else:
        df = df0
    if only_original:
        df = df[df.index==df['originalindex']]
    if validation:
        df = df[df['validation']]
    else:
        df = df[~df['validation']]
    with open(cfile, 'w') as fp:
        for l2 in df.itertuples():
            uta1 = l2.uta1
            if no_kugiri:
                uta1 = uta1.replace('−', '')
            fp.write(f'{uta1}\n\n')

def load_volume(dirname):
    return pd.read_csv(Path(dirname)/'volume.csv', index_col=0).fillna({'type': ''})
    
def load_all(dirname, vecname, c):
    dir = Path(dirname)
    if vecname is None or c is None:
        vec = None
    else:
        with h5py.File(dir/f'{c}'/'vec.h5', 'r') as infh:
            if vecname=='vec_mean0':
                n = np.array(infh['vec_n'])
                vec = ((np.array(infh[vec_cls])+(np.array(infh[vec_mean]).T*n).T+np.array(infh[vec_eos])).T/(n+2)).T
            else:
                vec = np.array(infh[vecname])
    volume_df = load_volume(dirname)
    verse_df = pd.read_csv(dir/'verse.csv', index_col=0)
    meta_df = verse_df[(verse_df.index==verse_df['originalindex']) & ~verse_df['validation']]
    meta_df = meta_df.drop(['uta1', 'dindex', 'validation'], axis=1)
    meta_df = meta_df.reset_index(drop=True)
    return vec, volume_df, verse_df, meta_df

def get_all_imperial_anthologies(dirname):
    volume_df = load_volume(dirname)
    return list(volume_df[volume_df['type'].str.contains('E')].index)

def title2code(dirname, title):
    volume_df = load_volume(dirname)
    return volume_df[volume_df['title']==title].index[0]

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, nx.classes.digraph.DiGraph):
            return nx.node_link_data(obj)
        return super(NpEncoder, self).default(obj)

def save_variables(dirname, gs, otherdata, fname='dump.json'):
    with open(Path(dirname)/fname, 'w') as fp:
        json.dump({'gs': gs, 'otherdata': otherdata}, fp, cls=NpEncoder)

def object_hook(d):
    return {int(k) if k.lstrip('-').isdigit() else k:
            nx.node_link_graph(v) if k=='g' else v
            for k, v in d.items()}
    
def load_variables(dirname, fname='dump.json'):
    with open(Path(dirname)/fname, 'r') as fp:
        json_dict = json.load(fp, object_hook=object_hook)
    return json_dict['gs'], json_dict['otherdata']
