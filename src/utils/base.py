#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from loss import PixelPrototypeCELoss
import torch.nn.functional as F
import torch.nn as nn

types_pretty = {'train': 'training', 'valid': 'validation', 'valid': 'valid'}

def metrics(predictions, gts, label_values=["Building", "Non-building"]):
    #predictions = gts = ndarray（2621440，）
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))

    accuracy = cm[1][1]/(cm[1][1]+cm[0][1]+cm[1][0]+1e-10)  # Iou
    oa = (cm[0][0] + cm[1][1]) / (cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]+1e-10)
    return accuracy, oa

def weighted_average_updates(w, n_k, weights):
    w_avg = deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key]*weights[0]*len(w), n_k[0])
        for i in range(1, len(w)):
            w_avg[key] = torch.add(w_avg[key], w[i][key]*weights[i]*len(w), alpha=n_k[i])
        w_avg[key] = torch.div(w_avg[key], sum(n_k))
    return w_avg

def inference(model, loader, device):
    if loader is None:
        return None, None

    criterion = PixelPrototypeCELoss().to(device)
    loss, total, correct = 0., 0, 0
    model.eval()
    
    with torch.no_grad():
        all_preds = []
        all_gts = []
        for batch, (examples, labels) in enumerate(loader):
            examples, labels = examples.to(device), labels.to(device)
            log_probs, _ = model(examples,labels)
            loss += criterion(log_probs, labels).item() * labels.shape[0]

            out_seg = log_probs['seg']
            _ , pred = torch.max(out_seg, dim=1,keepdim=True)
            pred = pred.detach().cpu().numpy().astype(np.float32)
            labels = labels.detach().cpu().numpy().astype(np.float32)

            all_preds.append(pred)
            all_gts.append(labels)
            total += labels.shape[0]
            
    accuracy, oa = metrics(np.concatenate([p.ravel() for p in all_preds]),
        np.concatenate([p.ravel() for p in all_gts]).ravel())
    # accuracy = correct/total
    loss /= total

    return accuracy, loss

def get_acc_avg(acc_types, clients, model, device):
    acc_avg = {}
    for type in acc_types:
        acc_avg[type] = 0.
        num_examples = 0
        for client_id in range(len(clients)):
            acc_client, _ = clients[client_id].inference(model, type=type, device=device)
            if acc_client is not None:
                acc_avg[type] += acc_client * len(clients[client_id].loaders[type].dataset)
                num_examples += len(clients[client_id].loaders[type].dataset)
        acc_avg[type] = acc_avg[type] / num_examples if num_examples != 0 else None
        # acc_avg[type] = acc_avg[type] /
    return acc_avg

def printlog_stats(quiet, logger, loss_avg, acc_avg, acc_types, lr, round, iter, iters):
    if not quiet:
        print(f'        Iteration: {iter}', end='')
        if iters is not None: print(f'/{iters}', end='')
        print()
        print(f'        Learning rate: {lr}')
        print(f'        Average running loss: {loss_avg:.6f}')
        for type in acc_types:
            print(f'        Average {types_pretty[type]} accuracy: {acc_avg[type]:.3%}')

    if logger is not None:
        logger.add_scalar('Learning rate (Round)', lr, round)
        logger.add_scalar('Learning rate (Iteration)', lr, iter)
        logger.add_scalar('Average running loss (Round)', loss_avg, round)
        logger.add_scalar('Average running loss (Iteration)', loss_avg, iter)
        for type in acc_types:
            logger.add_scalars('Average accuracy (Round)', {types_pretty[type].capitalize(): acc_avg[type]}, round)
            logger.add_scalars('Average accuracy (Iteration)', {types_pretty[type].capitalize(): acc_avg[type]}, iter)
        logger.flush()

def get_weights(local_protos):
    global_protos = torch.stack(local_protos)
    global_protos = torch.mean(global_protos, dim=0)
    cosine_similarities = []

    for tensor in local_protos:
        cosine_similarity = F.cosine_similarity(tensor.flatten(), global_protos.flatten(), dim=0)
        cosine_similarities.append(cosine_similarity.item())

    sum_of_similarities = sum(cosine_similarities)
    weights_list = [similarity / sum_of_similarities for similarity in cosine_similarities]

    return weights_list

def set_clients(args,Client):
    clients = []

    if args.dataset == "inria":
        clients_list = ["austin", "chicago", "kitsap", "tyrol-w", "vienna"]
        dataset_path = "../dataset/data-inria/"
    if args.dataset == "bhpools":
        clients_list = ["REGION_1", "REGION_2", "REGION_3", "REGION_4", "REGION_5", "REGION_6", "REGION_7", "REGION_8"]
        dataset_path = "../dataset/data-bhpools/"
    if args.dataset == "glm":
        clients_list = ["A_Luoi_Vietnam", "Chimanimani_Zimbabwe", "Kurucasile_Turkey", "Osh_Kyrgyzstan", "Tbilisi_Georgia","Tenejapa_Mexico"]
        dataset_path = "../dataset/data-GLM/"

    for client_id in clients_list:
        clients.append(Client(args=args, path_root=dataset_path, client_name=client_id))
    return clients

def aggregate_parameters(args,model,updates, num_examples, weights_list):
    update_avg = weighted_average_updates(updates, num_examples, weights_list)
    v = deepcopy(update_avg)
    for key in model.state_dict():
        if model.state_dict()[key].type() == v[key].type():
            model.state_dict()[key] -= v[key] * args.server_lr


class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

    def forward(self, x):
        out, x = self.base(x)
        out = self.head(out)

        return out, x