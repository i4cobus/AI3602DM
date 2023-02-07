import torch
import torch.nn as nn
from layers import Feature_Embedding, My_MLP


class NCF(nn.Module):
    def __init__(self, device, feature_dims, embed_size, hidden_nbs, user_field_idx, item_field_idx, dropout=0):
        super(NCF, self).__init__()
        self.user_field_idx = user_field_idx
        self.item_field_idx = item_field_idx
        self.embedding = Feature_Embedding(feature_dims=feature_dims, embed_size=embed_size, device=device)
        self.embed_out_dim = len(feature_dims) * embed_size
        self.mlp = My_MLP(input_dim=self.embed_out_dim,
                          hidden_nbs=hidden_nbs,
                          out_dim=hidden_nbs[-1], last_act=None, drop_rate=dropout)
        self.fc = nn.Linear(hidden_nbs[-1]+embed_size, 1)

    def forward(self, data):
        # 得到embeddign之后，reshape成一个向量
        data = self.embedding(data)
        # MF部分(要考虑batch部分，所以:表示取所有的batch数据)
        user_x = data[:, self.user_field_idx].squeeze(1)
        item_x = data[:, self.item_field_idx].squeeze(1)
        gmf = user_x * item_x
        # 多层感知机部分
        data = data.view(-1, self.embed_out_dim)
        out_mlp = self.mlp(data)
        # concat两部分
        data = torch.cat([gmf, out_mlp], dim=-1)
        out = torch.sigmoid(self.fc(data)).squeeze(1)
        return out
