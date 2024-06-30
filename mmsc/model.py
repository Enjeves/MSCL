import random
import torch
import torch.nn as nn
from typing import Callable

from mmsc.activations import RSoftmax, entmax15, t_softmax

class GatedFeatureLearningUnit(nn.Module):
    def __init__(
        self,
        n_features_in: int,
        n_stages: int,
        feature_mask_function: Callable = entmax15,
        feature_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features_in = n_features_in
        self.n_features_out = n_features_in
        self.feature_mask_function = feature_mask_function
        self._dropout = dropout
        self.n_stages = n_stages
        self.feature_sparsity = feature_sparsity
        self.learnable_sparsity = learnable_sparsity
        self._build_network()

    def _create_feature_mask(self):
        feature_masks = torch.cat(
            [
                torch.distributions.Beta(
                    torch.tensor([random.uniform(0.5, 10.0)]),
                    torch.tensor([random.uniform(0.5, 10.0)]),
                )
                .sample((self.n_features_in,))
                .squeeze(-1)
                for _ in range(self.n_stages)
            ]
        ).reshape(self.n_stages, self.n_features_in)
        return nn.Parameter(
            feature_masks,
            requires_grad=True,
        )

    def _build_network(self):
        self.W_in = nn.ModuleList(
            [nn.Linear(2 * self.n_features_in, 2 * self.n_features_in) for _ in range(self.n_stages)]
        )
        self.W_out = nn.ModuleList(
            [nn.Linear(2 * self.n_features_in, self.n_features_in) for _ in range(self.n_stages)]
        )

        self.feature_masks = self._create_feature_mask()
        if self.feature_mask_function.__name__ == "t_softmax":
            t = RSoftmax.calculate_t(self.feature_masks, r=torch.tensor([self.feature_sparsity]), dim=-1)
            self.t = nn.Parameter(t, requires_grad=self.learnable_sparsity)
        self.dropout = nn.Dropout(self._dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        t = torch.relu(self.t) if self.feature_mask_function.__name__ == "t_softmax" else None
        for d in range(self.n_stages):
            if self.feature_mask_function.__name__ == "t_softmax":
                feature = self.feature_mask_function(self.feature_masks[d], t[d]) * x
            else:
                feature = self.feature_mask_function(self.feature_masks[d]) * x
            h_in = self.W_in[d](torch.cat([feature, h], dim=-1))
            z = torch.sigmoid(h_in[:, : self.n_features_in])
            r = torch.sigmoid(h_in[:, self.n_features_in :])
            h_out = torch.tanh(self.W_out[d](torch.cat([r * h, x], dim=-1)))
            h = self.dropout((1 - z) * h + z * h_out)
        return h
   
    
class MLP(nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        super().__init__(*layers)


class MMSCARF(nn.Module):
    def __init__(
        self,
        input_dims,
        emb_dims,
        project_dim,
        encoder_depth=6,
        head_depth=2,
        dropout=0,
        dropout_poj=0,
        feature_sparsity=0.3,
        learnable_sparsity=True
    ):
        """Implementation of SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption.
        It consists in an encoder that learns the embeddings.
        It is done by minimizing the contrastive loss of a sample and a corrupted view of it.
        The corrupted view is built by remplacing a random set of features by another sample randomly drawn independently.

            Args:
                input_dim (int): size of the inputs
                emb_dim (int): dimension of the embedding space
                encoder_depth (int, optional): number of layers of the encoder MLP. Defaults to 4.
                head_depth (int, optional): number of layers of the pretraining head. Defaults to 2.
                corruption_rate (float, optional): fraction of features to corrupt. Defaults to 0.6.
                encoder (nn.Module, optional): encoder network to build the embeddings. Defaults to None.
                pretraining_head (nn.Module, optional): pretraining head for the training procedure. Defaults to None.
        """
        super().__init__()
        self.input_dims = input_dims
        for i in range(len(input_dims)):
            encoder_name = f'encoder{i}'
            #encoder = MLP(input_dims[i], emb_dims[i], encoder_depth, dropout)
            encoder = GatedFeatureLearningUnit(n_features_in=input_dims[i], n_stages=encoder_depth,
                                               feature_mask_function=t_softmax, dropout=dropout,
                                               feature_sparsity=feature_sparsity, learnable_sparsity= learnable_sparsity)
            self.add_module(encoder_name, encoder)
            
            encoder_name = f'project_head{i}'
            encoder = MLP(input_dims[i], project_dim, head_depth, dropout_poj)
            encoder.apply(self._init_weights)
            self.add_module(encoder_name, encoder)
        

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, Datas):
        emb_datas = []
        # Datas包含5个256*85的张量，data为单个256*85
        for i, data in enumerate(Datas):

            encoder = getattr(self, f'encoder{i}')
            pretraining_head = getattr(self, f'project_head{i}')
            
            emb_data = encoder(data)
            emb_data = pretraining_head(emb_data)
            emb_datas.append(emb_data)
        return emb_datas
    
    def get_embeddings(self, input):
        emb_datas = []
        for i, data in enumerate(input):
            encoder = getattr(self, f'encoder{i}')
            emb_data = encoder(data)
            emb_datas.append(emb_data)
        return emb_datas
    
    def feature_importance(self):
        feature_importance = []
        for i, _ in enumerate(self.input_dims):
            encoder = getattr(self, f'encoder{i}')
            improtance = encoder.feature_mask_function(encoder.feature_masks).sum(dim=0).detach().cpu().numpy()
            feature_importance.append(improtance)
            
        return feature_importance