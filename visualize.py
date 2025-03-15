import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import networkx as nx
import graphviz
import utilities

def main():
    dirnames = ['test1']
    symbols = ['^', 'v', '<', '>']
    if len(symbols)<len(dirnames):
        sys.exit('Supply more symbols for matplotlib.')
    
    gss, labels, counts, knn10s, knn23s, networknames = [], [], [], [], [], []
    for i, dirname in enumerate(dirnames):
        gs, (label, count, knn10, knn23) = utilities.load_variables(dirname)
        gss.append(gs)
        labels.append(label)
        counts.append(count)
        knn10s.append(knn10)
        knn23s.append(knn23)
        networknames.append(f'Model {i}')
    
    dirname0 = dirnames[0]
    label0 = labels[0]
    codes = utilities.get_all_imperial_anthologies(dirname0)+[0]
    figures = Path('figures')
    
    visualize_network_graph(dirname0, gss[0][0]['original']['g'], figures/'fig_network1.pdf')
    plot_average_pie_chart(label0, counts, figures/'fig_honkadori.pdf')
    plot_original_reverse_shuffle(gss, r'$\bar{r}$', figures/'fig_barr.pdf', ylim='log')
    plot_dict_values(dirname0, knn23s, networknames, symbols,
                     figures/'fig_knn23.pdf', open=[(utilities.title2code(dirname0, '古今集'),
                                                     utilities.title2code(dirname0, '新続古今集'))]*len(knn23s))
    plot_dict_values(dirname0, knn10s, networknames, symbols,
                     figures/'fig_knn10.pdf', open=[(utilities.title2code(dirname0, '古今集'),
                                                     utilities.title2code(dirname0, '新古今集'))]*len(knn10s))

    if len(dirnames)>1:
        congruence_between_networks = [get_congruence_between_network(dirname0, gs1, gs2, codes)
                                       for gs1, gs2 in itertools.combinations(gss, 2)]
        plot_degree_correlation(congruence_between_networks, codes, figures/'fig_congruence_among_networks.pdf')
    
    congruences = []
    predictions = []
    for congruence, prediction in [get_congruence_prediction(dirname0, gs, codes) for gs in gss]:
        congruences.append(congruence)
        predictions.append(prediction)
    
    plot_degree_correlation(congruences, codes, figures/'fig_congruences.pdf')
    plot_degree_correlation(predictions, codes, figures/'fig_predictions.pdf')
    
    chosen_nonchosen_after_list = [estimate_effect_being_chosen_all(dirname0, codes, gs) for gs in gss]
    plot_effect_being_chosen(chosen_nonchosen_after_list, codes, figures/'fig_effect_being_chosen.pdf')
    
def visualize_network_graph(dirname, g, fname):
    _, volume_df, verse_df, _ = utilities.load_all(dirname, None, None)
    verse2_df = verse_df[((verse_df['originalindex']==verse_df.index) & (~verse_df['validation']))]
    verse2_df.reset_index()
    root = 'root'
    size = '0.1'
    ag = graphviz.Digraph(engine='twopi')
    ag.attr('graph', root=root)
    ag.attr('node', shape='circle', label='', style='filled', width=size, height=size, penwidth='0')
    ag.attr('edge', arrowsize=size, penwidth=size, color='#808080')
    ag.node(root, '', color='transparent')
    cm = plt.get_cmap('Greens')
    def y2c(year):
        i = int(255*(0.25+(year-700)/(1600-700)*0.7))
        cc = '#'+''.join([f'{int(255*x):02x}' for x in cm(i)[:3]])
        return cc
    for n in g.nodes():
        year = volume_df.loc[verse2_df.iloc[n]['code'], 'chronology']
        ag.node(str(n), '', fillcolor=y2c(year))
    for n in g.nodes():
        if len(list(nx.ancestors(g, n)))==0:
            ag.edge(root, str(n), color='transparent')
        for n2 in g.successors(n):
            ag.edge(str(n), str(n2))
    ag.render(outfile=fname, cleanup=True)

def plot_average_pie_chart(labels, counts_list, fname):
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.pie(np.average(np.array(counts_list), axis=0), labels=labels, startangle=90,
           counterclock=False, autopct='%.1f%%', pctdistance=0.9, labeldistance=1.05,
           colors=sns.color_palette('Set2'))
    plt.savefig(fname)

def plot_original_reverse_shuffle(gss, ylabel, fname, ylim=None, panel=None, leftmargin=0.25):
    fig = plt.figure(figsize=(3, 2.5))
    plt.subplots_adjust(leftmargin, 0.2, 0.95, 0.9)
    average_rank_original = []
    average_rank_reverse = []
    average_rank_shuffle = []
    x = []
    xx = []
    for i, gs in enumerate(gss):
        average_rank_original.append(gs[0]['original']['av_rank'])
        average_rank_reverse.append(gs[0]['reverse']['av_rank'])
        average_rank_shuffle += [y['av_rank'] for y in gs[0]['shuffle']]
        x += [i+1]
        xx += [i+1]*(len(average_rank_shuffle)-len(xx))
    plt.scatter(x, average_rank_original, label='Real', marker='^', s=10, zorder=2)
    plt.scatter(x, average_rank_reverse, label='Reverse', marker='v', s=10, zorder=1)
    plt.scatter(xx, average_rank_shuffle, label='Shuffled', marker='x', s=10, zorder=0)
    plt.xlabel('Language model')
    plt.ylabel(ylabel)
    plt.xticks(x)
    plt.legend(handletextpad=0, frameon=False, loc='lower center',
               bbox_to_anchor=(0.4-(leftmargin-0.25), 1.02,), borderaxespad=0, ncol=3, columnspacing=1)
    if ylim=='log':
        plt.yscale('log')
    elif ylim is not None:
        plt.ylim(*ylim)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    if panel is not None:
        ax.text(0.025, 0.95, panel, va='center', transform=fig.transFigure, fontsize=12, weight='bold')
    plt.savefig(fname)
    
def get_intracollection_standization(g, meta_df, ind1, ind2, limit):
    def standardized_list(x):
        if len(x)<=1:
            return []
        else:
            std = np.std(x)
            if std==0:
                std = 1
            return (np.array(x)-np.mean(x))/std
    d_before = []
    d_after = []
    for i in range(ind1, ind2+1):
        s_before = s_after = 0
        for x in g.successors(i):
            if x>limit:
                s_after += 1
            else:
                s_before += 1
        d_before.append(s_before)
        d_after.append(s_after)
    return standardized_list(d_before), standardized_list(d_after)

def get_std_d_before_after(g, c, meta_df, volume_df):
    if c==0:
        limit = np.max(meta_df.index)
    else:
        limit = np.max(meta_df[meta_df['code']==c].index)
    std_d_before = []
    std_d_after = []
    for l in volume_df.itertuples():
        theindex = meta_df[meta_df['code']==l.Index].index
        if len(theindex)==0:
            continue
        ind1, ind2 = np.min(theindex), np.max(theindex)
        d_before, d_after = get_intracollection_standization(g, meta_df, ind1, ind2, limit)
        std_d_before += list(d_before)
        std_d_after += list(d_after)
        if l.Index==c:
            break
    return std_d_before, std_d_after

def get_congruence_prediction(dirname, gs, codes):
    _, volume_df, _, meta_df = utilities.load_all(dirname, None, None)
    congruence = {}
    prediction = {}
    for c in codes:
        if c==0:
            continue
        std_d0_before, std_d0_after = get_std_d_before_after(gs[0]['original']['g'], c, meta_df, volume_df)
        std_dc_before, _ = get_std_d_before_after(gs[c]['original']['g'], c, meta_df, volume_df)
        res_congruence = stats.spearmanr(std_d0_before, std_dc_before)
        res_prediction = stats.spearmanr(std_d0_after, std_dc_before)
        congruence[c] = (res_congruence.statistic, res_congruence.pvalue)
        prediction[c] = (res_prediction.statistic, res_prediction.pvalue)
    return congruence, prediction

def get_congruence_between_network(dirname, gs1, gs2, codes):
    _, volume_df, _, meta_df = utilities.load_all(dirname, None, None)
    congruence = {}
    for c in codes:
        std_d_before1, _ = get_std_d_before_after(gs1[c]['original']['g'], c, meta_df, volume_df)
        std_d_before2, _ = get_std_d_before_after(gs2[c]['original']['g'], c, meta_df, volume_df)
        res_congruence = stats.spearmanr(std_d_before1, std_d_before2)
        congruence[c] = (res_congruence.statistic, res_congruence.pvalue)
    return congruence

def plot_degree_correlation(results_list, codes, fname, xlabel='Phylogenetic network', panel=None):
    fig = plt.figure(figsize=(5.5, 3))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(0.15, 0.2, 0.95, 0.95)
    ticks = []
    for c in codes:
        if c not in results_list[0]:
            continue
        res = [res[c][0] for res in results_list]
        ax.scatter(c*np.ones_like(res), res, c='k', s=10, zorder=1)
        ticks.append(c)
        if c%2==0:
            ax.axvspan(c-0.5, c+0.5, color='lightgray', zorder=0)
    ax.set_xticks(ticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Correlation coefficient')
    ax.set_xlim(np.min(ticks)-0.5, np.max(ticks)+0.5)
    if panel is not None:
        ax.text(0.025, 0.95, panel, va='center', transform=fig.transFigure, fontsize=12, weight='bold')
    plt.savefig(fname)

def plot_dict_values(dirname, thedict_list, label_list, marker_list, fname, xlabel='Year of publication', panel=None, open=None):
    fig = plt.figure(figsize=(3, 2.5))
    plt.subplots_adjust(0.2, 0.2, 0.95, 0.95)
    volume_df = utilities.load_volume(dirname)
    cmap = plt.get_cmap('tab10')
    for i, (thedict, thelabel, themarker) in enumerate(zip(thedict_list, label_list, marker_list)):
        xy = np.array([(volume_df.loc[k, 'chronology'], v) for k, v in thedict.items()])
        if open is not None:
            openindex = [x in open[i] for x in thedict.keys()]
            nonopenindex = [x not in open[i] for x in thedict.keys()]
        else:
            openindex = np.arange(len(thedict))
        plt.scatter(xy[nonopenindex, 0], xy[nonopenindex, 1], label=thelabel, marker=themarker, s=10, color=cmap(i))
        if np.sum(np.array(openindex))>0:
            plt.scatter(xy[openindex, 0], xy[openindex, 1], marker=themarker, s=15, color=(1, 1, 1, 0), edgecolors=cmap(i))
    plt.legend(handletextpad=0, frameon=False)
    plt.xlabel(xlabel)
    plt.ylabel('Average score')
    if panel is not None:
        plt.text(0.025, 0.95, panel, va='center', transform=fig.transFigure, fontsize=12, weight='bold')
    plt.savefig(fname)

def estimate_effect_being_chosen(g0, gc, c, verse_df, meta_df):
    index_c = meta_df[meta_df['code']==c].index
    ind1_c, ind2_c = np.min(index_c), np.max(index_c)
    classic_index_c_df = verse_df[(verse_df['code']==c) & (verse_df.index!=verse_df['originalindex'])]
    copied_meta_df = meta_df.copy()
    copied_meta_df['vecindex'] = copied_meta_df.index
    merge_df = pd.merge(copied_meta_df, classic_index_c_df, on='originalindex', how='right')
    classic_index = list(merge_df['vecindex'])
    def count_successors_before_c(i):
        return len([x for x in gc.successors(i) if x<ind1_c])
    def count_successors_after_c(i):
        return len([x for x in g0.successors(i) if x>ind2_c])
    chosen_after = []
    non_chosen_after = []
    earlier_than_c = np.arange(ind1_c)
    sblist = np.array([count_successors_before_c(i) for i in earlier_than_c])
    for i in classic_index:
        if i>=ind1_c:
            continue
        code_i = meta_df.loc[i, 'code']
        index_i = meta_df[meta_df['code']==code_i].index
        ind1_i, ind2_i = np.min(index_i), np.max(index_i)
        population = np.where((earlier_than_c>=ind1_i) & (earlier_than_c<=ind2_i) & (sblist==count_successors_before_c(i)))[0]
        population = list(set(population)-set(classic_index))
        if len(population)>0:
            chosen_after.append(count_successors_after_c(i))
            non_chosen_after.append(np.average([count_successors_after_c(k) for k in population]))
    return np.average(chosen_after), np.average(non_chosen_after)

def estimate_effect_being_chosen_all(dirname, codes, gss):
    _, _, verse_df, meta_df = utilities.load_all(dirname, None, None)
    return {c: estimate_effect_being_chosen(gss[0]['original']['g'], gss[c]['original']['g'],
                                            c, verse_df, meta_df)
            for c in codes if c!=0}

class log_formatter(ticker.LogFormatter):
    def _num_to_string(self, x, vmin, vmax):
        if x>10000:
            s = '%1.0e'%x
        elif x<1 and x>=0.001:
            s = '%g'%x
        elif x<0.001:
            s = '%1.0e'%x
        else:
            s = self._pprint_val(x, vmax - vmin)
        return s
    
def plot_effect_being_chosen(chosen_nonchosen_after_list, codes, fname, panel=None):
    fig = plt.figure(figsize=(5.5, 3))
    ax = fig.add_subplot(111)
    plt.subplots_adjust(0.15, 0.2, 0.95, 0.95)
    ticks = []
    chosen = []
    non_chosen = []
    for c in codes:
        if c not in chosen_nonchosen_after_list[0]:
            continue
        for chosen_nonchosen_after in chosen_nonchosen_after_list:
            ax.plot([c+0.25, c-0.25], chosen_nonchosen_after[c], c='k', linewidth=0.5, zorder=1)
            chosen.append((c+0.25, chosen_nonchosen_after[c][0]))
            non_chosen.append((c-0.25, chosen_nonchosen_after[c][1]))
        ticks.append(c)
        if c%2==0:
            ax.axvspan(c-0.5, c+0.5, color='lightgray', zorder=0)
    chosen = np.array(chosen)
    non_chosen = np.array(non_chosen)
    ax.scatter(non_chosen[:, 0], non_chosen[:, 1], marker='v', s=10, label='Non-selected', zorder=2)
    ax.scatter(chosen[:, 0], chosen[:, 1], marker='^', s=10, label='Selected', zorder=2)
    ax.set_xticks(ticks)
    ax.set_xlabel('Imperial Anthology')
    ax.set_ylabel('Average number of children')
    ax.set_yscale('log')
    ax.set_xlim(np.min(ticks)-0.5, np.max(ticks)+0.5)
    formatter = log_formatter(labelOnlyBase=False, minor_thresholds=(2, 0.4))
    ax.get_yaxis().set_major_formatter(formatter)
    ax.get_yaxis().set_minor_formatter(formatter)
    ax.legend(handletextpad=0, frameon=True, facecolor='white', framealpha=1, edgecolor='white')
    plt.savefig(fname)

if __name__=='__main__':
    main()
