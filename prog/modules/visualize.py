from matplotlib import pyplot as plt
from ..modules.utils import *

def map_pred(X, y):
    preds = X.view(-1).detach().numpy()
    fig, ax = plt.subplots()
    ax.plot(preds, 'k')
    ax.grid()
    ax.margins(0)
    for target in y:
        ax.axvspan(target[0], target[1], facecolor='red', alpha=0.5)  
    plt.ylim(0, 1)
    plt.xlim(0, len(preds))
    plt.show()

def map_preds(report, trx_orfs, ensembl_trx, preds, n_display=5): 
    n=0
    for idx, trx in enumerate(report.keys()):
        n+=1
        if trx in trx_orfs:
            print('biotype:', ensembl_trx[trx]['biotype'])
            print('gene:', ensembl_trx[trx]['gene_name'])
            print('chromosome:', ensembl_trx[trx]['chromosome'])
            print(trx)
            print(report[trx])
            targets = []
            orfs = dict()
            for orf, attrs in trx_orfs[trx].items():
                if orf.startswith('ENSP'):
                    targets.append((attrs['start'], attrs['stop']))
                orfs[orf] = {'start':attrs['start'],
                            'stop':attrs['stop'],
                            'MS':attrs['MS'],
                            'TE':attrs['TE'],
                            'frame':attrs['frame']}
            orfs = {k:v for k,v in sorted(orfs.items(), key=lambda item:item[1]['start'])}
            for key, value in orfs.items():
                print(key, ' : ', value)
        map_pred(preds[idx], targets)
        if n==n_display:
            break