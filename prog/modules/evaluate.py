#imports
from torch import cat
import torch
from matplotlib import pyplot as plt
from torch import cat
from sklearn import metrics
from torchmetrics import PrecisionRecallCurve
from sklearn.metrics import auc
#project specific imports
from modules.utils import orf_retrieval, pad_seqs

def bin_pred(output, thresh):
    bin_pred = (output>thresh).int()
    return bin_pred

def iou_score(target, output):
    target = target.flatten()
    output = output.flatten()
    intersection = (target * output).sum()
    union = target.sum() + output.sum() - intersection
    iou = intersection / union
    return iou.item()

def recall_score(target, output):
    target = target.flatten()
    output = output.flatten()
    true_positives = ((target == 1) & (output == 1)).sum()
    false_negatives = ((target == 1) & (output == 0)).sum()
    recall = true_positives / (true_positives + false_negatives + 1e-7)
    return recall.item()

def get_preds(model, X_test):
    preds = []
    model.eval()
    for X in X_test:
        pad = torch.zeros(4,1000)
        X = torch.cat([pad,X,pad],dim=1).view(1,4,-1).cuda()
        out = model(X).view(-1)
        out = out[pad.shape[1]:-pad.shape[1]].cpu().detach()
        preds.append(out)
    return preds

def get_xFOMO(model, X_test, y_test, trxps, batch_size, w_size):
    model.eval()
    trx_xscores = dict()
    w_size = w_size
    batch_size = batch_size
    for X,y,trx in zip(X_test,y_test,trxps):
        xscores = []
        masked_X = []
        masked_y = []
        pad = torch.zeros(4,1000)
        pad_length, X_length = pad.shape[-1], X.shape[-1]
        X = torch.cat([pad,X,pad],dim=1).view(4,-1)
        for i in range(pad_length,pad_length+X_length):
            X_, y_ = X.clone().T, y.clone()
            X_[i:i+w_size] = torch.tensor([0.,0.,0.,0.])
            y_[i:i+w_size] = 0
            masked_X.append(X_.T)
            masked_y.append(y)
        for i in range(0, len(masked_X), batch_size):
            batch = masked_X[i:i+batch_size]
            batch_y = masked_y[i:i+batch_size]
            size = len(batch)
            batch = pad_seqs(batch, 4, min_pad=0).cuda()
            batch = batch.view(size, 4, -1)
            outputs = model(batch).view(size,1,-1)
            for out, y in zip(outputs, batch_y):
                out = out.flatten()
                out = out[pad_length:-pad_length].cpu().detach()
                pred = bin_pred(out, 0.5)
                iou = iou_score(y, pred)
                xscores.append(iou)
        trx_xscores[trx] = xscores
        print(len(trx_xscores))
    return trx_xscores


def get_orfs(preds, seqs_test, trxps):
    orfs = dict()
    for idx, out in enumerate(preds):
        trx = trxps[idx]
        seq_test = seqs_test[idx]
        orfs[trx] = orf_retrieval(seq_test, out.numpy(), t = 0.5, w_size = 10, cds_cov = 0.75)
    return orfs


def ROC_curve(list_preds, list_targets):
    #mean curve
    cat_preds = cat([x.flatten() for y in list_preds for x in y]).detach().numpy()
    cat_targets = cat([x.flatten() for y in list_targets for x in y]).long().detach().numpy()
    fpr, tpr, _ = metrics.roc_curve(cat_targets, cat_preds)
    auc_roc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color = 'black', linewidth=1)
    #individual curves
    for preds, targets in zip(list_preds, list_targets):
        cat_preds = cat([x.flatten() for x in preds]).detach().numpy()
        cat_targets = cat([x.flatten() for x in targets]).long().detach().numpy()
        fpr, tpr, _ = metrics.roc_curve(cat_targets, cat_preds)
        plt.plot(fpr, tpr, color = 'green', linewidth=0.5)
    
    plt.ylim(0.95, 1.01), plt.xlim(0, 1)
    plt.xlabel("False positive rate"), plt.ylabel("True positive rate"), plt.title('ROC curve')
    plt.legend(['ROC auc: {}'.format(round(auc_roc, 3))])
    plt.savefig('roc_curve.png')
    plt.show()
    plt.clf()


def PR_curve(list_preds, list_targets):
    pr_curve = PrecisionRecallCurve(task='binary')
    #mean curve
    cat_preds = cat([x.flatten() for y in list_preds for x in y])
    cat_targets = cat([x.flatten() for y in list_targets for x in y]).long()
    precision, recall, _ = pr_curve(cat_preds, cat_targets)
    auc_pr = auc(recall, precision)
    plt.plot(recall, precision, color = 'black', linewidth=1)
    #individual curves
    for preds, targets in zip(list_preds, list_targets):
        cat_preds = cat([x.flatten() for x in preds])
        cat_targets = cat([x.flatten() for x in targets]).long()
        precision, recall, _ = pr_curve(cat_preds, cat_targets)
        plt.plot(recall, precision, color = 'green', linewidth=0.5)
    
    plt.ylim(0.95, 1.01), plt.xlim(0, 1)
    plt.xlabel("Recall"), plt.ylabel("Precision"), plt.title('PR curve')
    plt.legend(['PR auc: {}'.format(round(auc_pr, 3))])
    plt.savefig('pr_curve.png')
    plt.show()
    plt.clf()