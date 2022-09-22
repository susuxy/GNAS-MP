import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.cell_train import Cell
from models.operations import OPS
from models.networks import MLP
from data import TransInput, TransOutput
from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch.conv import GATv2Conv as GATConv

class GNNCell(torch.nn.Module):
    def __init__(self, args, i):
        super(GNNCell, self).__init__()
        self.args = args
        self.i = 1
        input_dim = args.in_dim_V if self.i == 0 else args.node_dim
        self.conv = GraphConv(input_dim, args.node_dim)
        if self.args.batchnorm_op:
            self.batchnorm_V    = nn.BatchNorm1d(args.node_dim)
        else:
            self.batchnorm_V = None
        self.activate       = nn.LeakyReLU(args.leaky_slope)

    def forward(self, input):
        G, V_in = input['G'], input['V']
        V = V_in
        V = F.dropout(V, self.args.dropout, training = self.training)
        V = self.conv(G, V)
        if self.batchnorm_V:
            V = self.batchnorm_V(V)

        V = self.activate(V)
        V = V + V_in if self.i != 0 else V
        return { 'G' : G, 'V' : V }

class Model_Train(nn.Module):
    
    def __init__(self, args, genotypes, trans_input_fn, loss_fn):
        super().__init__()
        self.args           = args
        self.nb_layers      = args.nb_layers
        self.genotypes      = genotypes
        #self.cells          = nn.ModuleList([Cell(args, genotypes[i]) for i in range(self.nb_layers)])
        self.cells          = nn.ModuleList([GNNCell(args, i) for i in range(self.nb_layers)])
        self.trans_input    = TransInput(trans_input_fn)
        self.trans_output   = TransOutput(args)
        if args.pos_encode > 0:
            self.position_encoding = nn.Linear(args.pos_encode, args.node_dim)
        print(self)


    def forward(self, input):
        input = self.trans_input(input)
        G, V  = input['G'], input['V']
        if self.args.pos_encode > 0:
            V = V + self.position_encoding(G.ndata['pos_enc'].float().to("cuda"))
        output = {'G': G, 'V': V}
        for cell in self.cells:
            output = cell(output)
        output = self.trans_output(output)
        output = F.log_softmax(output, dim=1)
        return output