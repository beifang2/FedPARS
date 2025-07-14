#!/usr/bin/env python
# -*- coding: utf-8 -*-
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from utils.base import inference
from dataloader import Dataset
from loss import PixelPrototypeCELoss

class Client(object):
    def __init__(self, args, path_root, client_name):
        self.args = args
        self.name = client_name

        # Create dataloaders
        self.train_bs = self.args.train_bs 
        self.loaders = {}
        self.loaders['train'] = DataLoader(Dataset(path_root=path_root,mode="train",client_name=client_name), batch_size=self.train_bs, shuffle=True)
        self.loaders['valid'] = DataLoader(Dataset(path_root=path_root,mode="valid",client_name=client_name), batch_size=self.train_bs, shuffle=True)

        self.criterion = PixelPrototypeCELoss()

    def train(self, model, optim, device, global_protos):
        epochs = self.args.epochs
        train_loader = self.loaders['train']
        client_stats_every = self.args.client_stats_every if self.args.client_stats_every > 0 and self.args.client_stats_every < len(train_loader) else len(train_loader)

        model.train()
        model.to(device)
        self.criterion.to(device)
        model.train()
        model_server = deepcopy(model)
        iter = 0
        for epoch in range(epochs):
            agg_protos_label = {}
            loss_sum, loss_num_images, num_images = 0., 0, 0
            for batch, (examples, label_g) in enumerate(train_loader):
                
                examples, labels = examples.to(device), label_g.to(device)
                model.zero_grad()
                log_probs, prototype = model(examples,labels)
                loss = self.criterion(log_probs, labels)

                loss_sum += loss.item() * labels.shape[0]
                loss_num_images += labels.shape[0]
                num_images += labels.shape[0]

                loss.backward()
                optim.step()

                # After client_stats_every batches...
                if (batch + 1) % client_stats_every == 0:
                    # ...Compute average loss
                    loss_running = loss_sum / loss_num_images

                    # ...Print stats
                    if not self.args.quiet:
                        print('            ' + f'Epoch: {epoch+1}/{epochs}, '\
                                               f'Batch: {batch+1}/{len(train_loader)} (Image: {num_images}/{len(train_loader.dataset)}), '\
                                               f'Loss: {loss.item():.6f}, ' \
                                               f'Running loss: {loss_running:.6f}')

                    loss_sum, loss_num_images = 0., 0

                iter += 1

        # Compute model update
        model_update = {}
        # model.state_dict() PyTorch用于获取模型参数的方法，允许对模型的参数进行检查、保存和加载
        for key in model.state_dict():
            model_update[key] = torch.sub(model_server.state_dict()[key], model.state_dict()[key])

        # agg_protos_label为当前客户端最新的所有批次的原型数据（每一个批次对应一个原型）
        return model_update, len(train_loader.dataset), iter, loss_running, prototype

    def inference(self, model, type, device):
        return inference(model, self.loaders[type], device)
