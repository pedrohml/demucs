from typing import Iterator
from torch.nn.parameter import Parameter
from demucs.htdemucs import HTDemucs
from demucs.htdemucs_adapter import HTDemucsAdapter
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class HTDemucsClassifier(HTDemucsAdapter):
  def __init__(self, *args, **kwargs):
    self.n_classes = kwargs.pop('n_classes', 2)
    self.classifier_transformer_dim = kwargs.pop('classifier_transformer_dim', 168)
    self.cross_encode_output_channels = kwargs.pop('cross_encode_output_channels', 384)
    super().__init__(*args, **kwargs)

    encoder_layer = nn.TransformerEncoderLayer(d_model=self.classifier_transformer_dim, nhead=8, batch_first=True)
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

    self.classifier = nn.Sequential(
      nn.Linear(self.classifier_transformer_dim * self.cross_encode_output_channels, self.n_classes ** 2),
      nn.ReLU(),
      nn.LayerNorm((self.n_classes ** 2,)),
      nn.Linear(self.n_classes ** 2, self.n_classes),
    )

    self.set_train_mode('all')

  def set_train_mode(self, mode='all'):
    assert mode in ['all', 'only_classifier']
    self.train_mode = mode

  @staticmethod
  def from_htdemucs_weights(htmodel: nn.Module, *args, **kwargs):
    assert isinstance(htmodel, HTDemucs)
    new_model = HTDemucsClassifier(*args, *htmodel._init_args_kwargs[0], **kwargs, **htmodel._init_args_kwargs[1])
    new_model.load_htdemucs_weights(htmodel)
    return new_model

  def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
    if self.train_mode == 'all':
      return super().parameters(recurse)
    else:
      return itertools.chain(self.transformer.parameters(), self.classifier.parameters())

  def forward(self, mix):
    hidden_state_specs, _, hidden_state_times, _ = self.cross_encode(mix)

    x = hidden_state_specs[-1]
    xt = hidden_state_times[-1]

    B, C, _ = xt.shape

    assert C == self.cross_encode_output_channels, 'output from cross encode block doesn\'t match'

    x = torch.cat([x.view(B, C, -1), xt.view(B, C, 1, -1).repeat(1, 1, 2, 1).view(B, C, -1)], dim=-1)
    x = F.adaptive_max_pool1d(x, self.classifier_transformer_dim)

    x = self.transformer(x).view(B, -1)
    x = self.classifier(x)
    return F.softmax(x, dim=-1)