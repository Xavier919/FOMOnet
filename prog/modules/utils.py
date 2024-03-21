import torch
import numpy as np
from itertools import groupby
from operator import itemgetter
import logomaker
import pandas as pd
from sklearn.cluster import DBSCAN
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  

def pad_seqs(seqs, num_chan, min_pad=100):
    pad_seqs = []
    max_len = max([x.shape[1] for x in seqs])+min_pad
    for seq in seqs:
        diff_len = max_len - seq.shape[1]
        padL, padR = torch.zeros(num_chan, diff_len//2), torch.zeros(num_chan, diff_len//2+diff_len%2)
        pad_seq = torch.cat([padL, seq, padR], dim=1)
        pad_seqs.append(pad_seq)
    return torch.stack(pad_seqs, dim=0)

def utility_fct(Xy):
    seq1, seq2 = zip(*Xy)
    X, y = pad_seqs(seq1, 4), pad_seqs(seq2, 1)
    return (X, y)

def get_loss(X, y, out, loss_fct):
    loss = loss_fct(out, y).cuda()
    zero_mask = torch.all(X == 0, dim=1)
    zero_mask = zero_mask.unsqueeze(1)
    zero_mask = zero_mask.expand(-1, 1, -1)
    loss[zero_mask] = 0.
    lens = torch.sum(X, dim=(1,-1))
    loss_sums = torch.sum(loss, dim=(1,-1))
    return (loss_sums/lens).mean()

def map_seq(seq):
    mapping = {'N':[0.,0.,0.,0.], 'A':[1.,0.,0.,0.], 'T':[0.,1.,0.,0.], 'G':[0.,0.,1.,0.], 'C':[0.,0.,0.,1.]}
    return torch.tensor([mapping[x] for x in seq]).T

def find_orfs(seq, long=True, nc=False):
    start_codons, stop_codons = ['ATG'], ['TGA', 'TAA', 'TAG']
    if nc: start_codons = ['ATG', 'TTG', 'GTG', 'CTG']
    frames = [0,1,2]
    orfs = []
    for frame in frames:
        starts, stops = [], []
        for idx in list(range(frame, len(seq), 3)):
            codon = seq[idx:idx+3]
            if codon in start_codons: 
                starts.append(idx)
            elif codon in stop_codons:
                stops.append(idx+3)
        stops = stops[::-1]
        for idx, stop in enumerate(stops):
            for start in starts:
                if stop - start < 90 or any(i > start for i in stops[idx+1:]):
                    continue
                else:
                    orfs.append((start, stop))
                    if long == True:
                        break
    orfs = sorted(orfs, key=lambda x: x[0])
    return orfs

def check_drop(w, t, edge):
    if edge:
        return np.all(w) >= t
    else:
        return t <= np.max(w)-np.min(w)

def get_window(out, idx, w_size):
    edge = False
    if idx < w_size:
        edge = True
        return edge, out[:idx+w_size+3]
    elif idx+w_size+3 > len(out):
        edge = True
        return edge, out[idx-w_size:]
    else:
        return edge, out[idx-w_size:idx+w_size+3]

def valid_start(start, stops, idx):
    return any(i > start for i in stops[idx+1:])

def orf_retrieval(seq, out, t = 0.5, w_size = 7, cds_cov = 0.75):
    start_codons, stop_codons = ['ATG','TTG','GTG','CTG'], ['TGA','TAG','TAA']
    cds = []
    seq_len = len(seq)
    for frame in range(3):
        stops = [i for i in range(frame, seq_len, 3) if seq[i:i+3] in stop_codons][::-1]
        for idx, stop in enumerate(stops):
            e, w = get_window(out, stop, w_size)
            starts = [i for i in range(stop-3,-1,-3) if seq[i:i+3] in start_codons]
            if len(starts) == 0 or not check_drop(w, t, e):
                continue
            best_codon, best_codon_idx, best_cds_cov = None, None, 0
            for start in starts:
                e, w = get_window(out, start, w_size)
                if valid_start(start, stops, idx) or not check_drop(w[::-1], t, e) or stop - start < 90:
                    continue
                cov = np.sum(out[start:stop] >= t)/(stop-start)
                if best_codon == None or cov > best_cds_cov:
                    best_codon, best_codon_idx, best_cds_cov = seq[start:start+3], start, cov
            if best_codon != None and best_cds_cov >= cds_cov:
                cds.append((best_codon_idx, stop+3))
    return cds

def build_fasta(data, filename):
    trx_seqs = []
    for trx, seq in data:
        header = f'>{trx}'
        trx_seq = header + '\n' + seq.upper()
        trx_seqs.append(trx_seq)
    fasta_text = '\n'.join(trx_seqs)
    with open(filename + '.fa', 'w') as fasta_file:
        fasta_file.write(fasta_text)

def tag_fomo_orfs(trx_orfs, orfs):
    for trx, orfs_ in trx_orfs.items():
        for orf, attrs in orfs_.items():
            if trx not in orfs:
                attrs['fomonet'] = False
                attrs['fomonet_start'] = None
            for start, stop in orfs[trx]:
                if attrs['stop'] == stop:
                    attrs['fomonet'] = True
                    attrs['fomonet_start'] = start
            if attrs['stop'] not in [x[1] for x in orfs[trx]]:
                attrs['fomonet'] = False
                attrs['fomonet_start'] = None
    return trx_orfs

def xfomo(iou_list, seq, cds_start, cds_stop, min_motif_len=20, min_delta=0.02):
    median = np.median(iou_list)
    idx_arr = np.where(iou_list < median-min_delta)[0]
    grps_idx = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(idx_arr), lambda x: x[0]-x[1])]
    results = dict()
    for grp in grps_idx:
        rng = 9
        low_bnd, high_bnd = grp[0]-rng, grp[-1]+rng
        motif = seq[low_bnd:high_bnd+1]
        if len(motif) < min_motif_len:
            continue
        start_dist, stop_dist = (low_bnd+rng)-cds_start, (low_bnd+rng)-cds_stop
        if motif not in results:
            results[motif] = {'start_dist': [start_dist],
                              'stop_dist': [stop_dist]} 
        else:
            results[motif]['start_dist'].append(start_dist)
            results[motif]['stop_dist'].append(stop_dist)
    return results

def xfomo(iou_list, seq, cds_start, cds_stop, min_delta=0.1):
    median = np.median(iou_list)
    idx_arr = np.where(iou_list < median-min_delta)[0]
    grps_idx = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(idx_arr), lambda x: x[0]-x[1])]
    results = []
    for grp in grps_idx:
        rng = 10
        low_bnd, high_bnd = grp[0]-rng, grp[-1]+rng
        motif = seq[low_bnd:high_bnd+1]
        start_dist, stop_dist = (low_bnd+rng)-cds_start, (low_bnd+rng)-cds_stop
        rel_pos = (low_bnd+high_bnd//2)/len(iou_list)
        start_dists = [x[1] for x in results]
        if any(start_dist in range(x-5,x+5) for x in start_dists):
            continue
        results.append((motif, start_dist, stop_dist, rel_pos))
    return results

def apply_dbscan(motifs, eps, min_samples, ngram_range=(5, 5)):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range, binary=False)
    X = vectorizer.fit_transform(motifs)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    clusters = dbscan.fit_predict(X)
    clusters_dict = defaultdict(list)
    for cluster, motif in zip(clusters, motifs):
        clusters_dict[cluster].append(motif)
    pca = PCA(n_components=3)
    X_reduced_3d = pca.fit_transform(X.toarray())
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    unique_clusters = np.unique(clusters)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    for cluster, color in zip(unique_clusters, colors):
        if cluster == -1:
            continue
        indices = clusters == cluster
        ax.scatter(X_reduced_3d[indices, 0], X_reduced_3d[indices, 1], X_reduced_3d[indices, 2], color=color, label=f'Cluster {cluster}', s=10)
    ax.set_title('3D visualisation of DBSCAN clustering')
    ax.set_xlabel('PCA Feature 1')
    ax.set_ylabel('PCA Feature 2')
    ax.set_zlabel('PCA Feature 3')
    plt.legend()
    plt.savefig('dbscan_cluster_3d.png')
    plt.show()
    return clusters, motifs, clusters_dict

def get_logo(cluster_dict, cluster_id, path):
    motifs = cluster_dict
    counts_df = pd.DataFrame(columns=['A', 'C', 'G', 'T', 'N'])
    for motif in motifs:
        for i, nucleotide in enumerate(motif):
            if i >= len(counts_df):
                counts_df.loc[i] = [0, 0, 0, 0, 0]
            counts_df.loc[i, nucleotide] += 1
    freq_df = counts_df.div(counts_df.sum(axis=1), axis=0)
    logo = logomaker.Logo(freq_df, color_scheme='classic')
    logo.ax.set_ylabel('Frequency')
    logo.ax.set_title(f'Cluster #{cluster_id} consensus motif')
    plt.savefig(path)