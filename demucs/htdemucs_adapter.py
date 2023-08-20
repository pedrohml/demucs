    
from einops import rearrange
from fractions import Fraction
import torch
import torch.nn as nn
import torch.nn.functional as F
from demucs.hdemucs import HDecLayer, HEncLayer, MultiWrap, ScaledEmbedding
from demucs.htdemucs import HTDemucs

from typing import List, Tuple, Union

class HLayerBlock(nn.Module):
    def __init__(self,
                 layer_classes: List[nn.Module],
                 chins: Union[List[int], Tuple[int]],
                 chouts: Union[List[int], Tuple[int]],
                 kernel_size: int = 8,
                 stride: int = 4,
                 norm_groups: int = 1,
                 dconv: int = 1,
                 empty: int = False,
                 freq: bool = True,
                 norm: bool = True,
                 context: int = 0,
                 dconv_kw: dict = {},
                 pad: bool = True,
                 rewrite: bool = True,
                 # dconv mode
                 dconv_mode: int = 1,
                 dconv_depth: int = 2,
                 dconv_comp: int = 8,
                 dconv_init: float = 1e-3,
                 # Frequency branch
                 freq_bandwith: int = 2048,
                 multi = False,
                 multi_freqs = None,
                 freq_emb_weight: float = 0.2,
                 emb_scale: int = 10,
                 emb_smooth: bool = True):
        super().__init__()

        assert len(set([len(layer_classes), len(chins), len(chouts)])) == 1, 'layer classes, input channels and output channels must have the same size'

        self.depth = len(layer_classes)
        self.freq = freq
        self.freq_emb_weight = freq_emb_weight

        self.layers = nn.ModuleList([])

        layer_params = {
                "kernel_size": kernel_size,
                "stride": stride,
                "freq": self.freq,
                "pad": pad,
                "norm": norm,
                "rewrite": rewrite,
                "norm_groups": norm_groups,
                "empty": empty,
                "context": context,
                "dconv_kw": {
                    "depth": dconv_depth,
                    "compress": dconv_comp,
                    "init": dconv_init,
                    "gelu": True,
                },
            }

        if self.freq and not dconv:
            layer_params['last'] = False
        
        if self.freq and dconv and self.freq_emb_weight > 0.0:
            self.freq_emb = ScaledEmbedding(
                512, 48, smooth=emb_smooth, scale=emb_scale
            )
        else:
            self.freq_emb = None

        for idx, layer_class, chin, chout in zip(range(self.depth), layer_classes, chins, chouts):
            if idx == self.depth - 1 and 'last' in layer_params:
                layer_params['last'] = True
            block_layer = layer_class(chin, chout, dconv=dconv, **layer_params)
            if multi:
                block_layer = MultiWrap(block_layer, multi_freqs)
            self.layers.append(block_layer)

    def forward(self, x, encoder_hidden_states=None, encoder_lengths=None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if encoder_hidden_states is None:
            encoder_hidden_states = [None] * self.depth

        if encoder_lengths is None:
            encoder_lengths = [None] * self.depth

        assert len(encoder_hidden_states) == self.depth

        hidden_states = []
        lengths = []
        output = x

        for idx, layer, encoder_hidden_state, encoder_length in zip(range(self.depth), self.layers, encoder_hidden_states, encoder_lengths):
            lengths.append(output.shape[-1])

            if encoder_hidden_state is None:
                output = layer(output)
            else:
                output, _ = layer(output, encoder_hidden_state, encoder_length)

            if self.freq and idx == 0 and isinstance(layer, HEncLayer):
                frs = torch.arange(output.shape[-2], device=output.device)
                freq_embedding = self.freq_emb(frs).t()[None, :, :, None].expand_as(output)
                output = output + self.freq_emb_weight * freq_embedding

            hidden_states.append(output)

        return hidden_states, lengths

class HTDemucsAdapter(HTDemucs):
    def __init__(self, *args, **kwargs):
        super(HTDemucsAdapter, self).__init__(*args, **kwargs)

        audio_channels = kwargs['audio_channels']
        channels = kwargs['channels']
        depth = kwargs['depth']
        rewrite = kwargs['rewrite']
        norm_groups = kwargs['norm_groups']
        dconv_depth = kwargs['dconv_depth']
        dconv_comp = kwargs['dconv_comp']
        dconv_init = kwargs['dconv_init']

        chouts = [channels*2**g for g in range(depth)]
        cac_multipler = 2 if self.cac else 1

        block_params = {
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "pad": True,
            "norm": False,
            "rewrite": rewrite,
            "norm_groups": norm_groups,
            "dconv_kw": {
                "depth": dconv_depth,
                "compress": dconv_comp,
                "init": dconv_init,
                "gelu": True,
            },
        }

        encoder_block_chin = audio_channels
        self.encoder_block = HLayerBlock(
            [HEncLayer] * depth,
            [encoder_block_chin * cac_multipler] + chouts[:-1],
            chouts,
            dconv=1,
            freq=True,
            **block_params
        )
        self.tencoder_block = HLayerBlock(
            [HEncLayer] * depth,
            [encoder_block_chin] + chouts[:-1],
            chouts,
            dconv=1,
            freq=False,
            **block_params
        )

        block_params['context'] = 1

        decoder_block_chout = audio_channels * len(self.sources)
        self.decoder_block = HLayerBlock(
            [HDecLayer] * depth,
            list(reversed(chouts)),
            list(reversed(chouts))[1:] + [decoder_block_chout * cac_multipler],
            dconv=1,
            freq=True,
            **block_params
        )
        self.tdecoder_block = HLayerBlock(
            [HDecLayer] * depth,
            list(reversed(chouts)),
            list(reversed(chouts))[1:] + [decoder_block_chout],
            dconv=1,
            freq=False,
            **block_params
        )

    @staticmethod
    def from_htdemucs(htmodel: nn.Module):
        assert isinstance(htmodel, HTDemucs)
        new_model = HTDemucsAdapter(**htmodel._init_args_kwargs[1])
        new_model.load_state_dict(htmodel.state_dict(), strict=False)
        new_model.load_block_weights()
        return new_model
    
    def load_block_weights(self):
        # load first layer freq scaler
        self.encoder_block.freq_emb.load_state_dict(self.freq_emb.state_dict())

        for old_encoder, new_encoder in zip(self.encoder, self.encoder_block.layers):
            new_encoder.load_state_dict(old_encoder.state_dict())

        for old_decoder, new_decoder in zip(self.decoder, self.decoder_block.layers):
            new_decoder.load_state_dict(old_decoder.state_dict())

    def preprocess(self, mix):
        length_pre_pad = None
        if self.use_train_segment:
            if self.training:
                self.segment = Fraction(mix.shape[-1], self.samplerate)
            else:
                training_length = int(self.segment * self.samplerate)
                if mix.shape[-1] < training_length:
                    length_pre_pad = mix.shape[-1]
                    mix = F.pad(mix, (0, training_length - length_pre_pad))
        z = self._spec(mix)
        mag = self._magnitude(z).to(mix.device)
        x = mag

        # unlike previous Demucs, we always normalize because it is easier.
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        # x will be the freq. branch input.

        # Prepare the time branch input.
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        return x, xt, dict(mean=mean, std=std), dict(mean=meant, std=stdt), z

    def cross_encode(self, x_or_mix, xt=None):
        if xt is None:
            x, xt, *_ = self.preprocess(x_or_mix)
        else:
            x = x_or_mix

        hidden_states, lengths = self.encoder_block(x)
        hidden_states_t, lengths_t = self.tencoder_block(xt)

        x = hidden_states[-1]
        xt = hidden_states_t[-1]

        lengths.append(x)
        lengths_t.append(xt)

        if self.crosstransformer:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_upsampler_t(xt)

            x, xt = self.crosstransformer(x, xt)

            if self.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_downsampler_t(xt)

        hidden_states.append(x)
        hidden_states_t.append(xt)
    
        return hidden_states, lengths, hidden_states_t, lengths_t
    
    @property
    def training_length(self):
        return int(self.segment * self.samplerate)

    def forward(self, mix):
        length = mix.shape[-1]

        x, xt, x_scaler, xt_scaler, z = self.preprocess(mix)

        B, C, Fq, T = x.shape

        hidden_states, lengths, hidden_states_t, lengths_t = self.cross_encode(x, xt)

        # remove from list mid block values, so we can further apply skips connections correctly
        x = hidden_states.pop(-1)
        xt = hidden_states_t.pop(-1)
        lengths.pop(-1)
        lengths_t.pop(-1)

        hidden_states, *_ = self.decoder_block(x, hidden_states[::-1], lengths[::-1])
        hidden_states_t, *_ = self.tdecoder_block(xt, hidden_states_t[::-1], lengths_t[::-1])

        x = hidden_states[-1]
        xt = hidden_states_t[-1]

        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * x_scaler['std'][:, None] + x_scaler['mean'][:, None]

        # to cpu as mps doesnt support complex numbers
        # demucs issue #435 ##432
        # NOTE: in this case z already is on cpu
        # TODO: remove this when mps supports complex numbers
        x_is_mps = x.device.type == "mps"
        if x_is_mps:
            x = x.cpu()

        zout = self._mask(z, x)
        if self.use_train_segment:
            if self.training:
                x = self._ispec(zout, length)
            else:
                x = self._ispec(zout, self.training_length)
        else:
            x = self._ispec(zout, length)

        # back to mps device
        if x_is_mps:
            x = x.to("mps")

        if self.use_train_segment:
            if self.training:
                xt = xt.view(B, S, -1, length)
            else:
                xt = xt.view(B, S, -1, self.training_length)
        else:
            xt = xt.view(B, S, -1, length)
        xt = xt * xt_scaler['std'][:, None] + xt_scaler['mean'][:, None]
        x = xt + x
        if x.shape[-1] > length:
            x = x[..., :length]
        return x