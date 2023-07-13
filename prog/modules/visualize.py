from matplotlib import pyplot as plt
from modules.utils import *

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

def map_preds(report, trx_orfs, ensembl_trx, coordinates, n_display=5): 
    n=0
    for idx, trx in enumerate(report.keys()):
        n+=1
        print(trx)
        if trx in trx_orfs:
            print('recall:', report[trx]['recall']), print('iou:', report[trx]['iou'])
            print('biotype:', ensembl_trx[trx]['biotype'])
            print('gene:', ensembl_trx[trx]['gene_name'])
            print('predicted orfs:', coordinates[idx])
            orfs = dict()
            for orf, attrs in trx_orfs[trx].items():
                orfs[orf] = {'start':attrs['start'],
                            'stop':attrs['stop'],
                            'MS':attrs['MS'],
                            'TE':attrs['TE'],
                            'frame':attrs['frame']}
            orfs = {k:v for k,v in sorted(orfs.items(), key=lambda item:item[1]['start'])}
            for key, value in orfs.items():
                print(key, ' : ', value)
        map_pred(report[trx]['out'], coordinates[idx])
        if n==n_display:
            break

def map_preds_os(report, dataset, n_display=5):
    n=0
    trxps = [x for x in dataset.keys()]
    for trx, info in report.items():
        n+=1
        trx_ = trxps[trx]
        print(trx_), print('recall:', info['recall']), print('iou:', info['iou'])
        print(find_orfs(map_back(dataset[trx_]['mapped_seq'])))
        map_pred(info['out'], info['mapped_cds'])
        if n==n_display:
            break