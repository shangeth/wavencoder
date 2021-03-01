'''
Refer fairseq's wav2vec for more pretrained models
https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md

Code from fairseq's repository is used in this file
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import urllib.request
from argparse import Namespace
from tqdm import tqdm
from wavencoder.utils import _reporthook


from typing import Dict, List, Optional, Tuple
from torch import Tensor

class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)

def norm_block(is_layer_norm, dim, affine=True):
    if is_layer_norm:
        mod = nn.Sequential(
            TransposeLast(),
            Fp32LayerNorm(dim, elementwise_affine=affine),
            TransposeLast(),
        )
    else:
        mod = Fp32GroupNorm(1, dim, affine=affine)

    return mod

class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers,
        dropout,
        log_compression,
        skip_connections,
        residual_scale,
        non_affine_group_norm,
        activation,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
                nn.Dropout(p=dropout),
                norm_block(
                    is_layer_norm=False, dim=n_out, affine=not non_affine_group_norm
                ),
                activation,
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for dim, k, stride in conv_layers:
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

        self.log_compression = log_compression
        self.skip_connections = skip_connections
        self.residual_scale = math.sqrt(residual_scale)

    def forward(self, x):
        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            if self.skip_connections and x.size(1) == residual.size(1):
                tsz = x.size(2)
                r_tsz = residual.size(2)
                residual = residual[..., :: r_tsz // tsz][..., :tsz]
                x = (x + residual) * self.residual_scale

        if self.log_compression:
            x = x.abs()
            x = x + 1
            x = x.log()

        return x

class Wav2Vec(nn.Module):
    def __init__(self, pretrained=True, pretrained_path=None, device=torch.device("cpu")):
        """wav2vec model encoder with pretrained weights

        Example:
            import wavencoder
            model = wavencoder.model.Wav2Vec(pretrained=True)

        Args:
            pretrained (bool, optional): [Use pretrained weights, download if weights are not available already]. Defaults to True.
            pretrained_path ([type], optional): [path to pretrained weights, if already downloaded separately]. Defaults to None.
            device ([type], optional): [device(cpu/cuda) of the model]. Defaults to torch.device("cpu").

        Raises:
            Exception: None
        """
        super().__init__()
        self.device = device
        if pretrained:
            args = {'activation':'relu', 'no_progress_bar': False, 'log_interval': 500, 'log_format': 'json', 'seed': 1, 'fp16': True, 
            'fp16_init_scale': 128, 'fp16_scale_window': None, 'task': 'speech_pretraining', 'skip_invalid_size_inputs_valid_test': True, 
            'max_tokens': 1500000, 'max_sentences': None, 'num_workers': 6, 'train_subset': 'train', 'valid_subset': 'valid', 
            'max_sentences_valid': None, 'distributed_world_size': 16, 'distributed_rank': 0, 'distributed_backend': 'nccl', 
            'distributed_init_method': 'tcp://learnfair1521:31009', 'distributed_port': 31009, 'device_id': 0, 'ddp_backend': 'c10d', 
            'bucket_cap_mb': 150, 'fix_batches_to_gpus': False, 'arch': 'audio_cpc', 'criterion': 'binary_cross_entropy', 'max_epoch': 0, 
            'max_update': 400000, 'clip_norm': 25, 'sentence_avg': False, 'update_freq': [1], 'optimizer': 'adam', 'lr': [1e-06], 'momentum': 0.99, 
            'weight_decay': 0.0, 'lr_scheduler': 'cosine', 'lr_shrink': 0.1, 'min_lr': 1e-09, 'min_loss_scale': 0.0001, 
            'save_dir': '/checkpoint/abaevski/asr/libri/libri_2x_st2_more.fp16.fp32gn.mxup400000.lr1e-06.adam.lr0.005.cosine.ftl7.agl12.off=auto.skip_agg.res_scl0.5.log.warmup500.initlr1e-07.crt=binary_cross_entropy.negs=10.max_sz=150000.crp_bsz.mxtk1500000.uf1.sd1.ngpu16', 'restore_file': 'checkpoint_last.pt', 
            'reset_optimizer': False, 'reset_lr_scheduler': False, 'optimizer_overrides': '{}', 'save_interval': 1, 'save_interval_updates': 100000, 
            'keep_interval_updates': -1, 'no_save': False, 'no_epoch_checkpoints': True, 'validate_interval': 1, 'adam_betas': '(0.9, 0.999)', 
            'adam_eps': 1e-08, 'warmup_updates': 500, 'warmup_init_lr': 1e-07, 'max_lr': 0.005, 't_mult': 1, 'lr_period_updates': -1, 
            'data': '/private/home/abaevski/data/librispeech', 'sample_rate': 16000, 'resample_method': 'linear', 'max_sample_size': 150000, 
            'min_sample_size': None, 'bach_by_cropsize': True, 'fp32_group_norm': True, 
            'conv_feature_layers': '[(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1)]', 
            'conv_aggregator_layers': '[(512, 2, 1), (512, 3, 1), (512, 4, 1), (512, 5, 1), (512, 6, 1), (512, 7, 1), (512, 8, 1), (512, 9, 1), (512, 10, 1), (512, 11, 1), (512, 12, 1), (512, 13, 1)]', 
            'offset': 'auto', 'skip_connections_agg': True, 'residual_scale': 0.5, 'log_compression': True, 'num_negatives': 10, 'prediction_steps': 12, 
            'sample_distance': None, 'cross_sample_negatives': False, 'negative_sampling_seed': None, 'dropout': 0.0, 'dropout_features': 0.0, 'dropout_agg': 0.0, 
            'encoder': 'cnn', 'aggregator': 'cnn', 'td_learn_mode': 'fixed', 'td_features': 40, 'td_proj_dim': None, 'td_len': 25, 'td_stride': 10, 'td_no_log': False, 
            'td_no_preemp': False, 'td_no_mvn': False, 'td_norm_energy': False, 'td_linear_dim': None, 'skip_connections_feat': False, 'gru_dim': 512, 'layer_norm_before': False, 
            'no_group_norm': False, 'norm_location': 'default', 'no_conv_bias': False, 'agg_zero_pad': False, 'feature_preemp': False, 'instance_norm': False, 'abs_features': False, 
            'balanced_classes': False, 'project_features': 'none', 'non_affine_group_norm': False, 'layer_norm_before_feat': 0, 'layer_norm_after_feat': 9223372036854775807, 
            'layer_norm_before_agg': 0, 'layer_norm_after_agg': 9223372036854775807, 'latent_vars': None, 'latent_var_banks': 1, 'latent_temp': '1.0', 'latent_hard': False, 
            'predict_next': False, 'latent_no_gumbel': False, 'latent_norm': 'none', 'latent_init': 'xavier_normal', 'batch_by_cropsize': False}
        else:
            args = {'conv_feature_layers': '[(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2), (512, 1, 1), (512, 1, 1)]',
                    'activation':'relu',
                    'dropout': 0.0,
                    'log_compression':True,
                    'skip_connections_feat':False,
                    'residual_scale':0.5,
                    'non_affine_group_norm':False,
                    }    

        args = Namespace(**args)
        if args.activation == "relu":
            activation = nn.ReLU()
        elif args.activation == "gelu":
            activation = nn.GELU()
        else:
            raise Exception("unknown activation " + args.activation)

        feature_enc_layers = eval(args.conv_feature_layers)
        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=args.dropout,
            log_compression=args.log_compression,
            skip_connections=args.skip_connections_feat,
            residual_scale=args.residual_scale,
            non_affine_group_norm=args.non_affine_group_norm,
            activation=activation,
        )

        if pretrained:
            filename = "wav2vec_large.pt"
            pretrained_weights_link = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt"
            if pretrained_path == None:
                if not os.path.exists(filename):
                    print(f'Downloading the pretrained weights from fairseq({pretrained_weights_link}) ...', flush=True)
                    with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                        urllib.request.urlretrieve(pretrained_weights_link, filename, reporthook=_reporthook(t))
                cp = torch.load(filename, map_location=self.device)
            else: 
                cp = torch.load(pretrained_path, map_location=self.device)
            pretrained_dict = cp['model']
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            self.load_state_dict(model_dict)

    def forward(self, x):
        x = x.squeeze(1)
        return self.feature_extractor(x)