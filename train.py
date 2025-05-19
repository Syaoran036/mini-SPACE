#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thr April 17 17:06:12 2025

@author: TANG Qiming
"""

import torch

def adjust_learning_rate(init_lr, optimizer, iteration,seperation):
    lr = max(init_lr * (0.9 ** (iteration//seperation)), 0.0001)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    # print(f"learning rate adjusted from {init_lr} to {lr}.")
    return lr

def train_SPACE(model, train_data, optimizer, scheduler,
                device='cuda',
                a=0.5,
                loss_type='MSE',
                ):
        
    x, edge_index = train_data.x.to(torch.float).to(device), train_data.edge_index.to(torch.long).to(device) 
    optimizer.zero_grad()
        
    z, _ = model.encode(x, edge_index) 
    graph_loss = model.graph_loss(z, train_data.pos_edge_label_index) * a  
    loss = graph_loss  
        
    reconstructed_features = model.decoder_x(z)
    if loss_type=='BCE': 
        feature_loss = torch.nn.functional.binary_cross_entropy(reconstructed_features, x) * 10
    elif loss_type=='MSE':
        feature_loss = torch.nn.functional.mse_loss(reconstructed_features, x) * 10
        
    loss += feature_loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    return {'epoch_loss':loss.item(),
            'graph_loss':graph_loss.item(),
            'feature_loss':feature_loss.item()}
    

    
