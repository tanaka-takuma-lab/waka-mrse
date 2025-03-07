import collections
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import spatial
import networkx as nx
import utilities

def main():
    dirname = 'test1'
    vecname = vec_mean
    measure = mL2
    parents = 1
    reverse = True
    shuffle = 10
    
    codes = utilities.get_all_imperial_anthologies(dirname)+[0]
    gs = generate_graphs_all(dirname=dirname, vecname=vecname, codes=codes, parents=parents, reverse=reverse, shuffle=shuffle, measure=measure)

    label, count = shinkokin_honkadori_stat(dirname, gs[0]['original']['g'])

    knn10 = classify_LOOCV(dirname=dirname,
                           code0=utilities.title2code(dirname, '古今集'),
                           code1=utilities.title2code(dirname, '新古今集'),
                           vecname=vecname, codes=codes, measure=measure)
    knn23 = classify_LOOCV(dirname=dirname,
                           code0=utilities.title2code(dirname, '古今集'),
                           code1=utilities.title2code(dirname, '新続古今集'),
                           vecname=vecname, codes=codes, measure=measure)

    utilities.save_variables(dirname, gs, (label, count, knn10, knn23))

def get_index_of_closest(v_, n):
    v = np.array(v_)
    x = []
    for _ in range(n):
        x.append(np.argmin(v))
        v[x[-1]] = np.inf
    return np.array(x)

mL1 = 'L1'
mL2 = 'L2'
mcos = 'cos'

vec_mean = 'vec_mean'
vec_cls = 'vec_cls'
vec_eos = 'vec_eos'
vec_mean0 = 'vec_mean0'

def get_distance(v1, v2, measure):
    if measure==mL1:
        return spatial.distance.cdist(v1, v2, 'cityblock')
    elif measure==mL2:
        X = (-2*v2@v1.T+np.sum(v1**2, axis=1)).T+np.sum(v2**2, axis=1)
        X[X<0] = 0
        return np.sqrt(X)
    elif measure==mcos:
        return 1-(v1.T/np.linalg.norm(v1, axis=1)).T@(v2.T/np.linalg.norm(v2, axis=1))
    else:
        raise Exception('Unknown measure.')

def knn(x, c0, c1, k, measure, no_self=False):
    eps = 1e-10
    batchsize = 256
    y = []
    for start in range(0, len(x), batchsize):
        d0 = get_distance(x[start:start+batchsize], c0, measure)
        d1 = get_distance(x[start:start+batchsize], c1, measure)
        for v0, v1 in zip(d0, d1):
            if no_self:
                v0 = v0[v0>eps]
                v1 = v1[v1>eps]
            y.append(np.sum([i for _, i in
                             sorted(list(zip(v0, [-1]*len(v0)))+list(zip(v1, [1]*len(v1))))[:k]]))
    return (np.array(y)>0).astype(np.int32)

def classify_LOOCV(dirname, code0, code1, vecname, codes, measure, klist=np.arange(1, 10, 2)):
    vec, volume_df, _, meta_df = utilities.load_all(dirname, vecname, 0)
    vec0 = vec[meta_df[meta_df['code']==code0].index]
    vec1 = vec[meta_df[meta_df['code']==code1].index]
    accuracy_k = []
    for k in klist:
        y0 = knn(vec0, vec0, vec1, k, measure, True)
        TN, FP = np.sum(y0==0), np.sum(y0==1)
        y1 = knn(vec1, vec0, vec1, k, measure, True)
        FN, TP = np.sum(y1==0), np.sum(y1==1)
        accuracy = (TN+TP)/(TN+FP+FN+TP)
        accuracy_k.append((accuracy, k))
    best_accuracy, best_k = sorted(accuracy_k)[-1]
    print(f'Best k={best_k} (accuracy={best_accuracy}).')
    return {c: np.average(knn(vec[meta_df[meta_df['code']==c].index], vec0, vec1, best_k, measure, True))
            for c in codes if c!=0}

def shinkokin_honkadori_stat(dirname, g, verbose=True):
    _, volume_df, verse_df, meta_df = utilities.load_all(dirname, None, None)
    shinkokin_code = utilities.title2code(dirname, '新古今集')
    shinkokin_df = verse_df[verse_df['code']==shinkokin_code]
    g_undirected = g.to_undirected()
    dir = Path(dirname)
    honka_df = pd.read_csv('shinkokin_honka.csv', comment='#')
    print(f'{len(honka_df["id"].unique())} poems with {len(honka_df)} originals')
    ind1 = np.min(meta_df[meta_df['code']==shinkokin_code].index.values)
    print(f'There are {ind1} poems before Shinkokinshu.')
    label = ['Parent', 'Ancestor', 'Close relative', 'Remote relative', 'Unconnected', 'Anachronism', 'Not found']
    parent, ancestor, close_relative, remote_relative, unconnected, anachronism, notfound = np.arange(len(label))
    honka_dict = {}
    doubleparent_dict = {}
    def get_id(honka_dict, k, v):
        if k not in honka_dict:
            return v
        if honka_dict[k]>v:
            return v
        return honka_dict[k]
    for l in honka_df.itertuples():
        uta1_df = shinkokin_df[shinkokin_df['uta1']==l.uta]
        if len(uta1_df)==0:
            print(f'Not found. {l.uta1}.')
            honka_dict[l.id] = get_id(honka_dict, l.id, notfound)
            continue
        x0 = meta_df[meta_df['originalindex']==uta1_df['originalindex'].values[0]]
        if len(x0)==0:
            print(f'Not found. {l.uta1}.')
            honka_dict[l.id] = get_id(honka_dict, l.id, notfound)
            continue
        x0 = x0.index[0]
        code0 = meta_df.loc[x0, 'code']
        year0 = volume_df.loc[code0, 'chronology']
        ind1 = np.min(meta_df[meta_df['code']==code0].index)
        originalindex_df = verse_df[verse_df['uta1']==l.uta1]
        if len(originalindex_df)==0:
            print(f'Not found. {l.uta1}.')
            honka_dict[l.id] = get_id(honka_dict, l.id, notfound)
            continue
        originalindex = originalindex_df['originalindex'].values[0]
        x1_df = meta_df[meta_df['originalindex']==originalindex]
        if len(x1_df)==0:
            print(f'Out of range. {l.uta1} ({volume_df.loc[originalindex_df["code"].values[0], "title"]}).')
            honka_dict[l.id] = get_id(honka_dict, l.id, notfound)
            continue
        x1 = x1_df.index[0]
        code1 = meta_df.loc[x1, 'code']
        year1 = volume_df.loc[code1, 'chronology']
        if np.isnan(year1):
            print('No year', l)
            honka_dict[l.id] = get_id(honka_dict, l.id, notfound)
            raise Exceptio('No year')
        if year1>year0 or x1>=ind1:
            print(f'Reversed or in the same collection. {volume_df.loc[code1, "title"]} ({year1}) {volume_df.loc[code0, "title"]} ({year0})')
            honka_dict[l.id] = get_id(honka_dict, l.id, anachronism)
            continue
        for x in nx.ancestors(g, x0):
            if x1==x:
                pl = nx.shortest_path_length(g_undirected, x0, x1)
                if pl==1:
                    honka_dict[l.id] = get_id(honka_dict, l.id, parent)
                else:
                    honka_dict[l.id] = get_id(honka_dict, l.id, ancestor)
                break
        else:
            try:
                pl = nx.shortest_path_length(g_undirected, x0, x1)
                if pl<=4:
                    honka_dict[l.id] = get_id(honka_dict, l.id, close_relative)
                else:
                    honka_dict[l.id] = get_id(honka_dict, l.id, remote_relative)
            except nx.NetworkXNoPath:
                honka_dict[l.id] = get_id(honka_dict, l.id, unconnected)
                p = list(g.predecessors(x0))[0]
                uta2 = verse_df.loc[meta_df.loc[p, 'originalindex'], 'uta1']
                if verbose:
                    print('Poem', l.uta)
                    print('Honka', l.uta1)
                    print('Parent', uta2)
    count = collections.Counter(honka_dict.values())
    return label, [count[i] for i in range(len(label))]

def get_ancestral_graph(vec, meta_df, volume_df, parents, measure, rankdistmode, batchsize=256):
    g = nx.DiGraph()
    g.add_nodes_from(meta_df.index.values)
    if rankdistmode:
        sum_rank = sum_dist = counter = 0
    for l in volume_df[~volume_df['chronology'].isna()].itertuples():
        edges = []
        index = list(meta_df[meta_df['code']==l.Index].index)
        if len(index)==0 or np.min(index)==0:
            continue
        ind1, ind2 = np.min(index), np.max(index)+1
        n, m, r = ind1, len(vec)-ind2, ind2-ind1
        for start in range(ind1, ind2, batchsize):
            if rankdistmode:
                dmatrix = get_distance(vec[start:min(start+batchsize, ind2)], vec, measure)
                dmatrix[:, ind1:ind2] = np.inf
            else:
                dmatrix = get_distance(vec[start:min(start+batchsize, ind2)], vec[:ind1], measure)
            for i, dvec in enumerate(dmatrix, start):
                ps = get_index_of_closest(dvec[:ind1], parents)
                edges += [(k, i) for k in ps]
                if rankdistmode:
                    d_closest = [dvec[p] for p in ps]
                    sorted_dvec = sorted(dvec[dvec<=np.max(d_closest)])
                    sum_rank += sum([sorted_dvec.index(d) for d in d_closest])/(m+n-1)
                    sum_dist += sum(d_closest)
        if rankdistmode:
            counter += r*parents
        g.add_edges_from(edges)
    if rankdistmode:
        print(sum_rank/counter, sum_dist/counter)
        return g, sum_rank/counter, sum_dist/counter
    else:
        return g, None, None

def place_bookwise(index, vec0, meta_df0):
    shuffle_meta_index = []
    for i in index:
        shuffle_meta_index += list(meta_df0[meta_df0['code']==i].index)
    meta_df = meta_df0.loc[shuffle_meta_index].reset_index(drop=True)
    vec = vec0[shuffle_meta_index]
    return vec, meta_df

def generate_graphs(vec0, meta_df0, volume_df, parents=1, original=True, reverse=True, shuffle=10, measure=mL2, rankdistmode=True):
    trained_index = np.array(volume_df[~volume_df['chronology'].isna()].index)
    gs = {'shuffle': []}
    if original:
        print('Original')
        index = trained_index
        g, av_rank, av_dist = get_ancestral_graph(vec0, meta_df0, volume_df, parents=parents, measure=measure, rankdistmode=rankdistmode)
        gs['original'] = {'index': index, 'g': g, 'av_rank': av_rank, 'av_dist': av_dist}
    if reverse:
        print('Reverse')
        index = trained_index[::-1]
        vec, meta_df = place_bookwise(index, vec0, meta_df0)
        g, av_rank, av_dist = get_ancestral_graph(vec, meta_df, volume_df, parents=parents, measure=measure, rankdistmode=rankdistmode)
        gs['reverse'] = {'index': index, 'g': g, 'av_rank': av_rank, 'av_dist': av_dist}
    for i in range(shuffle):
        print('Shuffle', i)
        index = np.random.permutation(trained_index)
        vec, meta_df = place_bookwise(index, vec0, meta_df0)
        g, av_rank, av_dist = get_ancestral_graph(vec, meta_df, volume_df, parents=parents, measure=measure, rankdistmode=rankdistmode)
        gs['shuffle'].append({'index': index, 'g': g, 'av_rank': av_rank, 'av_dist': av_dist})
    return gs

def generate_graphs_all(dirname, vecname, codes, parents, reverse, shuffle, measure):
    gs = {}
    for c in codes:
        vec, volume_df, _, meta_df = utilities.load_all(dirname, vecname, c)
        if c==0:
            print(c, 'Whole dataset')
            gs[c] = generate_graphs(vec, meta_df, volume_df, parents=parents, reverse=reverse, shuffle=shuffle, measure=measure, rankdistmode=True)
        else:
            print(c, volume_df.loc[c, 'title'])
            gs[c] = generate_graphs(vec, meta_df, volume_df, parents=parents, reverse=False, shuffle=0, measure=measure, rankdistmode=False)
    return gs

if __name__=='__main__':
    main()
