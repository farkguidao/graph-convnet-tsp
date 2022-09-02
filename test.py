
import os
import json
import argparse
import time

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt

import networkx as nx
from sklearn.utils.class_weight import compute_class_weight

from tensorboardX import SummaryWriter
from fastprogress import master_bar, progress_bar

# Remove warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from config import *
from utils.graph_utils import *
from utils.google_tsp_reader import GoogleTSPReader
from utils.plot_utils import *
from models.gcn_model import ResidualGatedGCNModel, SearchApproximator
from utils.model_utils import *


config_path='configs/default.json'
config = get_config(config_path)
print("Loaded {}:\n{}".format(config_path, config))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
if torch.cuda.is_available():
    print("CUDA available, using GPU ID {}".format(0))
    dtypeFloat = torch.cuda.FloatTensor
    dtypeLong = torch.cuda.LongTensor
    torch.cuda.manual_seed(1)
else:
    print("CUDA not available")
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)


def get_random_tsp_p(batch_size, num_nodes, num_dims, beam_size):
    #     返回tsp坐标点，随机生成的边概率，beamsearch搜索到的最小值。
    nodes_coord = torch.rand(batch_size, num_nodes, num_dims).type(dtypeFloat)
    edge_p = torch.rand(batch_size, num_nodes, num_nodes).type(dtypeFloat)
    edge_values = torch.sum((nodes_coord.unsqueeze(2) - nodes_coord.unsqueeze(1)) ** 2, dim=-1) ** 0.5
    bs_nodes = beamsearch_tour_nodes_shortest(
        edge_p, edge_values, beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type='p')
    tour_batch = batch_tour_len_nodes(edge_values, bs_nodes)
    #     mean_tour = mean_tour_len_nodes(edge_values,bs_nodes)
    return nodes_coord, edge_values, edge_p, tour_batch

net = nn.DataParallel(SearchApproximator(config,dtypeFloat, dtypeLong))
if torch.cuda.is_available():
        net.cuda()
print(net)

# Compute number of network parameters
nb_param = 0
for param in net.parameters():
    nb_param += np.prod(list(param.data.size()))
print('Number of parameters:', nb_param)


learning_rate = config.learning_rate
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
print(optimizer)

epoch_bar = master_bar(range(2000))
for batch_num in epoch_bar:
    net.train()
    nodes_coord, edge_values, edge_p, tour_batch = get_random_tsp_p(config.batch_size,config.num_nodes,config.node_dim,config.beam_size)
    tour_pre = net(nodes_coord,edge_p,edge_values)
    loss = nn.MSELoss()(tour_pre,tour_batch)
    optimizer.zero_grad()
    loss.backward()
    # Optimizer step
    optimizer.step()
    epoch_bar.write('t:'+ str(float(loss)))

