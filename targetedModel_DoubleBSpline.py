import argparse
import torch
import pickle as pkl
import torch.nn as nn
import utils as utils
import numpy as np
from modules import GCN, NN, Predictor, Discriminator, Density_Estimator, Discriminator_simplified, Truncated_power


class TR(nn.Module):
    def __init__(self, degree, knots):
        super(TR, self).__init__()
        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis
        self.weight = nn.Parameter(torch.rand(self.d), requires_grad=True)

    def forward(self, t):
        out = self.spb.forward(t)
        out = torch.matmul(out, self.weight)
        return out

    def _initialize_weights(self):
        # self.weight.data.normal_(0, 0.1)
        self.weight.data.zero_()


class TargetedModel_DoubleBSpline(nn.Module):

    def __init__(self, Xshape, hidden, dropout, num_grid=None, init_weight=True, tr_knots=0.25, cfg_density=None):
        super(TargetedModel_DoubleBSpline, self).__init__()
        if num_grid is None:
            num_grid = 20

        self.encoder = GCN(nfeat=Xshape, nclass=hidden, dropout=dropout)
        self.X_XN = Predictor(input_size=hidden + Xshape, hidden_size1=hidden, hidden_size2=hidden, output_size=int(hidden/2))
        self.Q1 = Predictor(input_size=int(hidden/2) + 1, hidden_size1=int(hidden*2), hidden_size2=hidden, output_size=1)
        self.Q0 = Predictor(input_size=int(hidden/2) + 1, hidden_size1=int(hidden*2), hidden_size2=hidden, output_size=1)
        self.g_T = Discriminator_simplified(input_size=int(hidden/2), hidden_size1=hidden, output_size=1)
        self.g_Z = Density_Estimator(input_size=int(hidden/2), num_grid=num_grid)
        tr_knots = list(np.arange(tr_knots, 1, tr_knots))
        tr_degree = 2
        self.tr_reg_t1 = TR(tr_degree, tr_knots)
        self.tr_reg_t0 = TR(tr_degree, tr_knots)

        if init_weight:
            self.encoder._initialize_weights()
            self.X_XN._initialize_weights()
            self.Q1._initialize_weights()
            self.Q0._initialize_weights()
            self.g_Z._initialize_weights()
            self.g_T._initialize_weights()
            self.tr_reg_t1._initialize_weights()
            self.tr_reg_t0._initialize_weights()

    def parameter_base(self):
        return list(self.encoder.parameters()) +\
            list(self.X_XN.parameters()) +\
            list(self.Q1.parameters())+list(self.Q0.parameters())+\
            list(self.g_T.parameters())+\
            list(self.g_Z.parameters())

    def parameter_targeted(self):
        return list(self.tr_reg_t0.parameters()) + list(self.tr_reg_t1.parameters())

    def tr_reg(self, T, neighborAverageT):
        tr_reg_t1 = self.tr_reg_t1(neighborAverageT)
        tr_reg_t0 = self.tr_reg_t0(neighborAverageT)
        regur = torch.where(T==1, tr_reg_t1, tr_reg_t0)
        return regur



    def forward(self, A, X, T, Z=None):
        embeddings = self.encoder(X, A)  # X_i,X_N
        embeddings = self.X_XN(torch.cat((embeddings,X), dim=1))

        g_T_hat = self.g_T(embeddings)  # X_i,X_N -> T_i
        if Z is None:
            neighbors = torch.sum(A, 1)
            neighborAverageT = torch.squeeze((A @ torch.unsqueeze(T, dim=1)), dim=1) / neighbors  # treated_neighbors / all_neighbors
        else:
            neighborAverageT = Z


        g_Z_hat = self.g_Z(embeddings, neighborAverageT)  # X_i,X_N -> Z
        g_Z_hat = torch.unsqueeze(g_Z_hat, dim=1)


        embed_avgT = torch.cat((embeddings, torch.unsqueeze(neighborAverageT, dim=1)), dim=1)

        Q_hat = torch.unsqueeze(T, dim=1) * self.Q1(embed_avgT) + (1-torch.unsqueeze(T, dim=1)) * self.Q0(embed_avgT)

        epsilon = self.tr_reg(T, neighborAverageT)  # epsilon(T,Z)


        return g_T_hat, g_Z_hat, Q_hat, torch.unsqueeze(epsilon, dim=1), embeddings, neighborAverageT

    def infer_potential_outcome(self, A, X, T, Z=None):
        embeddings = self.encoder(X, A)  # X_i,X_N
        embeddings = self.X_XN(torch.cat((embeddings,X), dim=1))

        g_T_hat = torch.squeeze(self.g_T(embeddings), dim=1)  # X_i,X_N -> T_i

        if Z is None:
            neighbors = torch.sum(A, 1)
            neighborAverageT = torch.squeeze((A @ torch.unsqueeze(T, dim=1)), dim=1) / neighbors  # treated_neighbors / all_neighbors
        else:
            neighborAverageT = Z


        g_Z_hat = self.g_Z(embeddings, neighborAverageT)  # X_i,X_N -> Z

        embed_avgT = torch.cat((embeddings, torch.unsqueeze(neighborAverageT, dim=1)), 1)

        Q_hat = torch.unsqueeze(T, dim=1) * self.Q1(embed_avgT) + (1-torch.unsqueeze(T, dim=1)) * self.Q0(embed_avgT)

        epsilon = self.tr_reg(T, neighborAverageT)  # epsilon(T,Z)
        # epsilon = epsilon.squeeze(1)


        return torch.squeeze(Q_hat, dim=1) + torch.unsqueeze(epsilon, dim=1) /(g_Z_hat * g_T_hat + 1e-6)

