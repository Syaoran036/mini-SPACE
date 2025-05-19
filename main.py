#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thr April 17 17:06:12 2025

@author: TANG Qiming
"""

import os
import random
import numpy as np
import scanpy as sc
import squidpy as sq
import scipy as sci
import networkx as nx
from sklearn.preprocessing import MaxAbsScaler

import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric import seed_everything

import matplotlib.pyplot as plt

from .layer import GAT_Encoder
from .model import SPACE_Graph
from .train import train_SPACE
from .utils import graph_construction

class EarlyStopping:
    """
    Early stops the training if loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, verbose=False, checkpoint_file=''):
        """
        Parameters
        ----------
        patience 
            How long to wait after last time loss improved. Default: 10
        verbose
            If True, prints a message for each loss improvement. Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.inf
        self.checkpoint_file = checkpoint_file

    def __call__(self, loss, model):
        # loss=loss.cpu().detach().numpy()
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter > self.patience:
                self.early_stop = True
                model.load_model(self.checkpoint_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''
        Saves model when loss decrease.
        '''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_file)
        self.loss_min = loss

def SPACE_GraphConstruct(adata,method='Squidpy',k=20,loss_type = 'MSE'):
    if method == 'SPACE':
        graph_dict = graph_construction(adata.obsm['spatial_fov'], adata.shape[0], k=k)
        adj = graph_dict.toarray()
    elif method == 'Squidpy':
        sq.gr.spatial_neighbors(
            adata,
            coord_type='generic',
            spatial_key='spatial_fov',
            delaunay=True
        )
        adj = adata.obsp['spatial_connectivities']
    
    G = nx.from_numpy_array(adj).to_undirected()
    edge_index = torch_geometric.utils.convert.from_networkx(G).edge_index

    if sci.sparse.issparse(adata.X):
        X_hvg = adata.X.toarray()
    else:
        X_hvg = adata.X.copy()

    if loss_type == 'BCE':
        scaler = MaxAbsScaler()
        scaled_x = torch.from_numpy(scaler.fit_transform(X_hvg))
    else:
        scaled_x = torch.from_numpy(X_hvg)

    data_obj = Data(edge_index=edge_index, x=scaled_x)
    data_obj.num_nodes = X_hvg.shape[0]
    data_obj.train_mask = data_obj.val_mask = data_obj.test_mask = data_obj.y = None

    return data_obj


def SPACE(adata,
          graph_const_method='Squidpy',
          epoch=1000,  # 总epoch数，这里我们理解为所有样本整体训练epoch数
          heads=6,
          k=20,
          alpha=0.5,
          lr=0.0005,
          patience=50,
          seed=42,
          GPU=True,
          outdir='./',
          loss_type='MSE',
          # save_attn=False,
          verbose=False,
):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    seed_everything(seed)

    os.makedirs(outdir, exist_ok=True)

    # 获取所有样本ID列表
    sample_ids = adata.obs['SampleID'].unique().tolist()

    # 训练模型初始化（共享模型），放在外面
    # 这里初始化encoder和model
    print('Setting up model ...')
    # 先用一个样本初始化模型参数维度等
    sample0_mask = adata.obs['SampleID'] == sample_ids[0]
    adata_sample0 = adata[sample0_mask].copy()
    
    # 构建图
    data_obj = SPACE_GraphConstruct(adata_sample0,
                                     method=graph_const_method,
                                     k=k,
                                     loss_type = loss_type)
    num_features = data_obj.num_features

    encoder = GAT_Encoder(
        in_channels=num_features,
        num_heads={'first': heads, 'second': heads, 'mean': heads},
        hidden_dims=[128, 128],
        dropout=[0.3, 0.3],
        concat={'first': True, 'second': True}
    )
    model = SPACE_Graph(encoder=encoder, decoder=None, loss_type=loss_type)

    # 设备选择
    if GPU and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    model.train()
    early_stopping = EarlyStopping(patience=patience, checkpoint_file=os.path.join(outdir,'model.pt'))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # 若想用 SGD + Momentum，改为
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # 例如学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(sample_ids), gamma=0.9)

    y_loss = {}  # loss history
    y_loss['feature_loss'] = []
    y_loss['graph_loss'] = []
    y_loss['epoch_loss']=[]
    x_epoch = []
    # 训练循环 - 每个epoch循环训练每个样本
    for epoch_i in range(epoch):
        x_epoch.append(epoch_i)
        loss2 = loss1 = epoch_loss = 0.0

        for sample_id in sample_ids:
            # 针对当前样本构建Data对象和图
            sample_mask = adata.obs['SampleID'] == sample_id
            adata_sample = adata[sample_mask].copy()

            data_obj = SPACE_GraphConstruct(adata_sample, 
                                            method=graph_const_method,
                                            k=k,
                                            loss_type = loss_type)

            transform = T.RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=True, 
                                  add_negative_train_samples=False, split_labels=True)
            train_data, _, _ = transform(data_obj)

            # 单样本训练
            single_loss = train_SPACE(model, train_data, optimizer, scheduler,
                                      device = device,
                                      a=alpha,
                                      loss_type=loss_type,
                                      )

            loss2 += single_loss['feature_loss']
            loss1 += single_loss['graph_loss']
            epoch_loss += single_loss['epoch_loss']
        
        loss2/=len(sample_ids)
        loss1/=len(sample_ids)
        epoch_loss /= len(sample_ids)

        y_loss['feature_loss'].append(loss2)
        y_loss['graph_loss'].append(loss1)
        y_loss['epoch_loss'].append(epoch_loss)

        if verbose and epoch_i%50==0:
            print('====> Epoch: {}, Loss: {:.4f}'.format(epoch_i, epoch_loss))

        early_stopping(epoch_loss, model,verbose=verbose,loss_type=loss_type)   
        if early_stopping.early_stop:
            print('====> Epoch: {}, Loss: {:.4f}'.format(epoch_i, epoch_loss)) 
            print('EarlyStopping: run {} iteration'.format(epoch_i))
            break

    fig = plt.figure()
    plt.plot(x_epoch, y_loss['epoch_loss'], 'go-', label='loss',linewidth=1, markersize=2)
    plt.plot(x_epoch, y_loss['graph_loss'], 'ro-', label='graph_loss',linewidth=1, markersize=2)
    plt.plot(x_epoch, y_loss['feature_loss'], 'bo-', label='feature_loss',linewidth=1, markersize=2)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    fig.savefig(os.path.join(outdir, 'train_loss.pdf'))

    print('Compute SPACE embeddings')
    # Load model
    pretrained_dict = torch.load(os.path.join(outdir,'model.pt'), map_location=device)                            
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)
    model = model.eval()

    # 字典保存每个样本的嵌入
    latent_embeddings = {}

    with torch.no_grad():
        for sample_id in sample_ids:
            sample_mask = adata.obs['SampleID'] == sample_id
            adata_sample = adata[sample_mask].copy()

            data_obj = SPACE_GraphConstruct(adata_sample, method=graph_const_method)

            x = data_obj.x.to(torch.float).to(device)
            edge_index_t = data_obj.edge_index.to(torch.long).to(device)
            z_nodes, attn_w = model.encode(x, edge_index_t)
            latent_embeddings[sample_id] = (adata_sample.obs_names.to_list(), z_nodes.cpu().numpy())

    # 合并所有样本嵌入到adata.obsm['latent']
    latent_all = np.zeros((adata.shape[0], list(latent_embeddings.values())[0][1].shape[1]))
    for sample_id, (cell_names, embeddings) in latent_embeddings.items():
        idxs = [adata.obs_names.get_loc(cell) for cell in cell_names]
        latent_all[idxs, :] = embeddings

    adata.obsm['latent'] = latent_all

    # 后续邻居计算与UMAP同之前
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=10, use_rep='latent', random_state=seed, key_added='SPACE')
    sc.tl.umap(adata, random_state=seed, neighbors_key='SPACE')

    # 保存结果
    adata.write(os.path.join(outdir, 'adata.h5ad'))

    return adata
