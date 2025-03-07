import numpy as np
import pandas as pd
import h5py
from pathlib import Path
import utilities
import analyze
import visualize

def main():
    d = 100
    tau = 5000
    alpha = 0.6
    utilities.safe_mkdir(figures)
    
    dummy_volume_df, dummy_verse_df, dummy_vec = generate_dummy_volume_verse_vec(num_volume, 1000, d, tau, alpha)
    save_dummy(dirname, dummy_volume_df, dummy_verse_df, dummy_vec)
    
    gs = analyze.generate_graphs_all(dirname=dirname, vecname=vecname, codes=[0], parents=parents, reverse=reverse, shuffle=shuffle, measure=measure)
    
    knn12 = analyze.classify_LOOCV(dirname=dirname, code0=1, code1=12,
                                   vecname=vecname, codes=codes, measure=measure)
    knn24 = analyze.classify_LOOCV(dirname=dirname, code0=1, code1=24,
                                   vecname=vecname, codes=codes, measure=measure)

    visualize.plot_original_reverse_shuffle([gs], r'$\bar{r}$', figures/'fig_model_barr.pdf', ylim='log')
    visualize.plot_dict_values(dirname, [knn12, knn24], ['1-12 classifier', '1-24 classifier'], ['^', 'v'],
                     figures/'fig_model_knn.pdf', xlabel='Anthology', open=[(1, 12), (1, 24)])
    copied_gs = {i: gs[0] for i in [0]+list(codes)}
    _, prediction = visualize.get_congruence_prediction(dirname, copied_gs, codes)

    visualize.plot_degree_correlation([prediction], codes, figures/'fig_model_predictions.pdf', xlabel='Anthology')

num_volume = 24
vecname = 'vec_mean'
dirname = 'test1'
measure = 'L2'
parents = 1
reverse = True
shuffle = 10
figures = Path('figures')
codes = np.arange(1, num_volume+1)

def generate_dynamics(n, d, tau, decay):
    x = np.zeros((1, d))
    weight = np.ones(1)
    n0 = int(10*tau)
    for i in range(1, n0+n):
        k = np.random.choice(i, size=1, p=weight/np.sum(weight))
        newx = decay*x[k]+np.random.normal(size=(1, d))
        weight = np.concatenate((np.exp(-1/tau)*weight, [1]))
        x = np.concatenate((x, newx), axis=0)
    return np.array(x[n0:])

def generate_dummy_volume_verse_vec(v, n_per_v, d1, tau, decay):
    volume_df = pd.DataFrame({'title': [str(i) for i in range(1, v+1)],
                              'chronology': np.arange(1, v+1),
                              'type': ['E']*v})
    volume_df = volume_df.set_index('chronology', drop=False)
    n = v*n_per_v
    verse_df = pd.DataFrame({'index': np.arange(n),
                             'originalindex': np.arange(n),
                             'dindex': np.arange(n),
                             'code': [i//n_per_v+1 for i in range(n)],
                             'chapter': ['' for _ in range(n)],
                             'uta1': ['' for _ in range(n)],
                             'author': ['' for _ in range(n)],
                             'validation': [False for _ in range(n)]})
    vec = generate_dynamics(n, d1, tau, decay)
    return volume_df, verse_df, vec

def save_dummy(dirname, volume_df, verse_df, vec):
    utilities.safe_mkdir(dirname)
    dir = Path(dirname)
    utilities.safe_mkdir(dir/'0')
    volume_df.to_csv(dir/'volume.csv', index=False)
    verse_df.to_csv(dir/'verse.csv', index=False)
    with h5py.File(dir/'0'/'vec.h5', 'w') as outfh:
        outfh.create_dataset(vecname, data=vec)

if __name__=='__main__':
    main()
