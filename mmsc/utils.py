import random

import numpy as np
import torch
from tqdm.auto import tqdm

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

def train_epoch(model, criterion, train_loader, optimizer, device, epoch):
    #启动训练模式
    model.train()

    epoch_loss = 0.0
    #progress bar
    batch = tqdm(train_loader, desc=f"Epoch {epoch}", leave = False)
    
    for Datas, label in batch:
        Datas, label = [data.to(device) for data in Datas], label.to(device)
        
        # reset gradients
        optimizer.zero_grad()

        # get embeddings
        emb_datas = model(Datas)
        
        # compute losss
        features = torch.cat([emb_data.unsqueeze(1) for emb_data in emb_datas], dim=1)
        # features = torch.stack([emb_dataa, emb_datab], dim=1)
        features = torch.nn.functional.normalize(features, dim=2, p=2)
        
        loss = criterion(features, label)
        #loss = criterion(features)
        loss.backward()

        # update model weights
        optimizer.step()

        # log progress
        epoch_loss += label.size(0) * loss.item()
        batch.set_postfix({"loss": loss.item()})

    return epoch_loss / len(train_loader.dataset)


def dataset_embeddings(model, loader, device):
    #评估模式
    model.eval()
    data_embedding = []
    data_embeddings = None
    with torch.no_grad():
        for Datas, _ in tqdm(loader):
            Datas = [data.to(device) for data in Datas]
            data_embedding = model.get_embeddings(Datas)
            if not data_embeddings:
                data_embeddings = [[] for _ in range(len(data_embedding))]
            for i in range(len(data_embedding)):
                data_embeddings[i].append(data_embedding[i])

    data_embeddings = [torch.cat(data).numpy() for data in data_embeddings]

    return data_embeddings


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def evaluate_model(test_target, vanilla_predictions, probs, the_labels, otfile):
    acc = accuracy_score(test_target, vanilla_predictions)
    print("ACC: " + str(acc), file=otfile)
    
    cm = confusion_matrix(test_target, vanilla_predictions, labels=the_labels)
    sen = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    spe = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    
    if len(np.unique(test_target)) == 2:
        auc = roc_auc_score(test_target, probs)
    else:
        auc = None
    print("SEN: ", sen, file=otfile)
    print("SPE: ", spe, file=otfile)
    print("AUC: ", auc, file=otfile)
    return acc, sen, spe, auc

def change_threshold(model, test_embeddings, test_target, rate, the_labels, otfile):
    thresholds = np.arange(-1.5, 1.5, 0.01)
    best_threshold = 0
    best_sen = 0
    best_spe = 0
    best_acc = 0

    for threshold in thresholds:

        confidence_scores = model.predict_proba(test_embeddings)

        predictions = np.where(confidence_scores[:,1] > threshold, the_labels[1], the_labels[0])

        tn, fp, fn, tp = confusion_matrix(test_target, predictions).ravel()
        current_sen = tp / (tp + fn)
        current_spe = tn / (tn + fp)
        current_acc = (tp + tn) / (tp + tn + fp + fn)
        

        if rate*current_acc + current_sen + current_spe > rate*best_acc + best_sen + best_spe:
        #if current_sen > best_sen and current_spe > best_spe:
            
            best_sen = current_sen
            best_spe = current_spe
            best_threshold = threshold
            best_acc = current_acc

    final_predictions = np.where(confidence_scores[:,1] > best_threshold, the_labels[1], the_labels[0])
    print("Best Threshold:", best_threshold, file=otfile)
    return final_predictions