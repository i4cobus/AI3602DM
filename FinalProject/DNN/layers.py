import torch
import numpy as np
import torch.nn as nn


class Feature_Embedding(nn.Module):
    '''
    实现特征的嵌入
    '''
    def __init__(self, feature_dims, embed_size, device):
        '''
        :param feature_dims: 各个特征的数量，如[3, 32, 343]表示特征1有3个取值范围，特征2有32个取值范围
        :param embed_size: 嵌入向量的维度，这里嵌入到同一个维度
        '''
        super(Feature_Embedding, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(sum(feature_dims), embedding_dim=embed_size)
        self.offset = np.array([0, *np.cumsum(feature_dims)[:-1]], dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, data):
        '''
        :param data: Long Tensor
        :return:
        '''
        # unsqueeze主要是考虑到batch的存在
        data = data + torch.tensor(self.offset, dtype=torch.long).unsqueeze(0).to(self.device)
        return self.embedding(data)


class Feature_Embedding_Sum(nn.Module):
    '''
    对特征向量化后，然后对所有向量求和，得到一个包含了所有信息的向量
    '''
    def __init__(self, feature_dims, device, out_dim=1):
        super(Feature_Embedding_Sum, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(sum(feature_dims), out_dim)
        self.bias = nn.Parameter(torch.zeros((out_dim,)))
        self.offset = np.array([0, *np.cumsum(feature_dims)[:-1]], dtype=np.long)

    def forward(self, data):
        '''
        :param data: Long Tensor
        :return:
        '''
        # unsqueeze主要是考虑到batch的存在
        data = data + torch.tensor(self.offset, dtype=torch.long).unsqueeze(0).to(self.device)
        # 把所有embedding之后的值向量叠加起来，得到一个向量
        data = torch.sum(self.embedding(data), dim=1) + self.bias
        return data


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_nbs, out_dim, last_act='sigmoid', drop_rate=0.2):
        '''
        :param input_dim: 输入层的神经元个数
        :param hidden_nbs: 列表，存储的是各个隐藏层神经元的个数
        :param out_dim: 输出层的维度
        :param last_act: 输出层的激活函数 'sigmoid', 'softmax'
        :param drop_rate:
        '''
        super(MLP, self).__init__()
        layers = []
        for nb in hidden_nbs:
            layers.append(nn.Linear(input_dim, nb))
            layers.append(nn.BatchNorm1d(nb))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=drop_rate))
            input_dim = nb
        layers.append(nn.Linear(input_dim, out_dim))
        self.mlp = nn.Sequential(*layers)
        if last_act == 'sigmoid':
            self.mlp.add_module('sigmoid', nn.Sigmoid())
        elif last_act == 'softmax':
            self.mlp.add_module('softmax', nn.Softmax())

    def forward(self, data):
        return self.mlp(data)


class My_MLP(nn.Module):
    def __init__(self, input_dim, hidden_nbs, out_dim, last_act='sigmoid', drop_rate=0.2):
        '''
        :param input_dim: 输入层的神经元个数
        :param hidden_nbs: 列表，存储的是各个隐藏层神经元的个数
        :param out_dim: 输出层的维度
        :param last_act: 输出层的激活函数 'sigmoid', 'softmax'
        :param drop_rate:
        '''
        super(My_MLP, self).__init__()
        layers = []
        for nb in hidden_nbs:
            layers.append(nn.Linear(input_dim, nb))
            layers.append(nn.BatchNorm1d(nb))
            layers.append(Poly_2(nb))
            layers.append(nn.Dropout(p=drop_rate))
            input_dim = nb
        layers.append(nn.Linear(input_dim, out_dim))
        self.mlp = nn.Sequential(*layers)
        if last_act == 'sigmoid':
            self.mlp.add_module('sigmoid', nn.Sigmoid())
        elif last_act == 'softmax':
            self.mlp.add_module('softmax', nn.Softmax())

    def forward(self, data):
        return self.mlp(data)


class FactorizationMachine(nn.Module):
    '''
    因子分解的部分
    '''
    def __init__(self, reduce_sum=True):
        super(FactorizationMachine, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, data):
        square_of_sum = torch.sum(data, dim=1) ** 2
        sum_of_square = torch.sum(data ** 2, dim=1)
        data = square_of_sum - sum_of_square
        if self.reduce_sum:
            data = torch.sum(data, dim=1, keepdim=True)
        return 0.5 * data


class CrossNet(nn.Module):
    def __init__(self, input_dim, num_layers):
        super(CrossNet, self).__init__()
        self.num_layer = num_layers
        self.w = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = nn.ParameterList(
            nn.Parameter(torch.zeros((input_dim, ))) for _ in range(num_layers)
        )

    def forward(self, x):
        x0 = x
        for i in range(self.num_layer):
            xw = self.w[i](x)
            x = x0*xw + self.b[i] + x
        return x


class Poly_2(nn.Module):
    '''
    形如： a*x^2 + b*x + c
    其中的a和b对每个神经元来说都是独一无二的
    '''
    def __init__(self, neuro_nb):
        super(Poly_2, self).__init__()
        self.a = nn.Parameter(torch.randn(neuro_nb, ))
        self.b = nn.Parameter(torch.randn(neuro_nb, ))
        self.c = nn.Parameter(torch.randn(neuro_nb, ))

    def forward(self, data):
        x = data * data
        x = self.a * x + self.b*data + self.c
        return x