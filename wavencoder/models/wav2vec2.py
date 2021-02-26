# '''
# Refer fairseq's wav2vec for more pretrained models
# https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md

# Code from fairseq's repository is used in this file
# '''
# from argparse import Namespace
# import math
# from dataclasses import dataclass, field
# from typing import List, Tuple

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# import fairseq
# from fairseq import utils
# from fairseq.data.data_utils import compute_mask_indices
# from fairseq.dataclass import ChoiceEnum, FairseqDataclass
# from fairseq.models import BaseFairseqModel, register_model
# from fairseq.modules import (
#     Fp32GroupNorm,
#     Fp32LayerNorm,
#     GradMultiply,
#     GumbelVectorQuantizer,
#     LayerNorm,
#     MultiheadAttention,
#     SamePad,
#     TransposeLast,
# )
# from fairseq.modules.transformer_sentence_encoder import init_bert_params
# from fairseq.utils import buffered_arange

# EXTRACTOR_MODE_CHOICES = ChoiceEnum(["default", "layer_norm"])
# MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])


# class Wav2Vec2Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         cfg = {'activation_dropout': 0.0,
#  'activation_fn': 'gelu',
#  'adam_betas': '(0.9,0.98)',
#  'adam_eps': 1e-06,
#  'arch': 'wav2vec2',
#  'attention_dropout': 0.1,
#  'attention_type': 'default',
#  'augment': False,
#  'best_checkpoint_metric': 'loss',
#  'bpe': None,
#  'bucket_cap_mb': 25,
#  'centroids': None,
#  'clip_norm': 25,
#  'codebook_negatives': 0,
#  'combine_banks': False,
#  'conv_bias': False,
#  'conv_feature_layers': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 2',
#  'conv_pos': 128,
#  'conv_pos_groups': 16,
#  'conv_pos_layers': 1,
#  'cpu': False,
#  'criterion': 'wav2vec',
#  'cross_sample_negatives': 0,
#  'curriculum': 0,
#  'data': '/private/home/abaevski/data/librispeech/full',
#  'dataset_impl': None,
#  'ddp_backend': 'c10d',
#  'device_id': 0,
#  'disable_validation': False,
#  'distributed_backend': 'nccl',
#  'distributed_init_method': 'tcp://learnfair2106:55498',
#  'distributed_no_spawn': True,
#  'distributed_port': 55498,
#  'distributed_rank': 0,
#  'distributed_world_size': 64,
#  'div_drop_percent': 0,
#  'div_pen_threshold': None,
#  'dropout': 0.1,
#  'dropout_features': 0.1,
#  'dropout_input': 0.1,
#  'duplicate_negatives': 0,
#  'empty_cache_freq': 0,
#  'enable_padding': False,
#  'encode_padded_indiv': False,
#  'encoder_attention_heads': 12,
#  'encoder_embed_dim': 768,
#  'encoder_ffn_embed_dim': 3072,
#  'encoder_layerdrop': 0.05,
#  'encoder_layers': 12,
#  'encoder_normalize_before': True,
#  'end_learning_rate': 0.0,
#  'extractor_model': None,
#  'extractor_norm_location': 'default',
#  'fast_stat_sync': False,
#  'feature_glu': False,
#  'feature_grad_mult': 0.1,
#  'feature_noise': 0.0,
#  'feature_noise_last': 0.0,
#  'features_pen': True,
#  'final_dim': 256,
#  'find_unused_parameters': True,
#  'finetune_extractor': True,
#  'fix_batches_to_gpus': False,
#  'fixed_validation_seed': None,
#  'force_anneal': None,
#  'fp16': True,
#  'fp16_init_scale': 128,
#  'fp16_scale_tolerance': 0.0,
#  'fp16_scale_window': None,
#  'group_norm_features': False,
#  'group_norm_groups': 512,
#  'gumbel_noise_gain': 1,
#  'infomax': True,
#  'input_noise': 0.0,
#  'keep_interval_updates': 1,
#  'keep_last_epochs': -1,
#  'label_smoothing': 0.0,
#  'labels': None,
#  'latent_groups': 2,
#  'latent_temp': '(2,0.5,0.999995)',
#  'latent_var_banks': 2,
#  'latent_vars': 320,
#  'layer_norm_after': 9223372036854775807,
#  'layer_norm_before': 0,
#  'layer_norm_features': True,
#  'lazy_load_labels': False,
#  'log_format': 'json',
#  'log_interval': 200,
#  'logit_temp': 0.1,
#  'loss_weights': None,
#  'lr': [0.0005],
#  'lr_scheduler': 'polynomial_decay',
#  'mask_min_space': 1,
#  'mask_multiple_length': 10,
#  'mask_prob': 0.65,
#  'mask_same_channels': False,
#  'mask_same_timesteps': False,
#  'mask_selection': 'static',
#  'mask_stdev': 0.0,
#  'masking_schedule': 0,
#  'max_epoch': 0,
#  'max_positions': 8000,
#  'max_pred_length': 0,
#  'max_sample_size': 250000,
#  'max_sentences': None,
#  'max_sentences_valid': None,
#  'max_tokens': 1400000,
#  'max_tokens_valid': 1400000,
#  'max_update': 400000,
#  'maximize_best_checkpoint_metric': False,
#  'memory_efficient_fp16': False,
#  'min_loss_scale': 0.0001,
#  'min_lr': -1,
#  'min_sample_size': 32000,
#  'mlp_mi': 256,
#  'negatives_from_everywhere': False,
#  'new_emb_pen': True,
#  'new_logit_pen': False,
#  'no_bert_init': False,
#  'no_epoch_checkpoints': True,
#  'no_last_checkpoints': False,
#  'no_mask_channel_overlap': False,
#  'no_mask_overlap': False,
#  'no_norm_after': 0,
#  'no_progress_bar': False,
#  'no_save': False,
#  'no_save_optimizer_state': False,
#  'no_token_positional_embeddings': True,
#  'noise_type': 'gaussian',
#  'norm_init_weight': 1.0,
#  'normalize': False,
#  'num_negatives': 100,
#  'num_workers': 6,
#  'optimizer': 'adam',
#  'optimizer_overrides': '{}',
#  'penalty_coeff': '[0, 0, 0.1, 10]',
#  'penalty_temp': 1.0,
#  'pooler_activation_fn': 'tanh',
#  'pooler_dropout': 0.0,
#  'power': 1.0,
#  'pre_norm': False,
#  'predictor_grad_mult': 1.0,
#  'preemp': False,
#  'project_quantized': True,
#  'quantize_input': False,
#  'quantize_targets': True,
#  'quantized': False,
#  'quantizer_chance': 0.0,
#  'quantizer_grad_mult': 1.0,
#  'quantizer_init': True,
#  'quantizer_init_gain': 1.0,
#  'quantizer_init_normal': True,
#  'relative_positional_embeddings': 0,
#  'required_batch_size_multiple': 8,
#  'resample_method': 'linear',
#  'rescale_sample_size': False,
#  'reset_dataloader': False,
#  'reset_lr_scheduler': False,
#  'reset_meters': False,
#  'reset_optimizer': False,
#  'restore_file': 'checkpoint_last.pt',
#  'same_quantizer': False,
#  'sample_rate': 16000,
#  'save_dir': '/checkpoint/abaevski/asr/speechbert_raw_250k_q5/best_ld_400k.nep.qtz.nnf0.ng512.pq.lv320.lvb2.lr0.0005.mask10.mprob0.65.mstd0.mstd0.05.drp_i0.1.drp_f0.1.fgm0.1.qini1.fpen.pen0_0_0.1_10.cpl1.neg100.mxsz250000.s5.ngpu64',
#  'save_interval': 1,
#  'save_interval_updates': 25000,
#  'scp': False,
#  'seed': 5,
#  'sentence_avg': False,
#  'siamese_extractor': False,
#  'siamese_feature_layers': None,
#  'skip_connections': False,
#  'skip_invalid_size_inputs_valid_test': True,
#  'skip_main_loss_prob': 0,
#  'soft': False,
#  'squeeze_constant': 20,
#  'squeeze_logits': 'norm_temp',
#  'squeeze_pos_emb': 'add',
#  'squeeze_quantizer_logits': False,
#  'static_preemp': False,
#  'tanh_after_norm': 0.0,
#  'task': 'audio_pretraining',
#  'tensorboard_logdir': '',
#  'threshold_loss_scale': None,
#  'tokenizer': None,
#  'total_num_update': 400000,
#  'train_on_full': False,
#  'train_subset': 'train',
#  'unprojected_feats': False,
#  'update_freq': [1],
#  'use_aggregator_feats': False,
#  'use_bmuf': False,
#  'use_old_adam': False,
#  'user_dir': None,
#  'valid_subset': 'valid',
#  'validate_after_updates': 0,
#  'validate_interval': 5,
#  'validate_interval_updates': 10000,
#  'warmup_updates': 32000,
#  'weight_decay': 0.01,
#  'weight_norm': False}

#         cfg = Namespace(**cfg)
#         self.cfg = cfg
#         feature_enc_layers = eval(cfg.conv_feature_layers)
#         self.embed = feature_enc_layers[-1][0]

#         self.feature_extractor = ConvFeatureExtractionModel(
#             conv_layers=feature_enc_layers,
#             dropout=0.0,
#             # mode=cfg.extractor_mode,
#             conv_bias=cfg.conv_bias,
#         )

#         self.post_extract_proj = (
#             nn.Linear(self.embed, cfg.encoder_embed_dim)
#             if self.embed != cfg.encoder_embed_dim and not cfg.quantize_input
#             else None
#         )

#         self.mask_prob = cfg.mask_prob
#         self.mask_selection = cfg.mask_selection
#         self.mask_other = cfg.mask_other
#         self.mask_length = cfg.mask_length
#         self.no_mask_overlap = cfg.no_mask_overlap
#         self.mask_min_space = cfg.mask_min_space

#         self.mask_channel_prob = cfg.mask_channel_prob
#         self.mask_channel_selection = cfg.mask_channel_selection
#         self.mask_channel_other = cfg.mask_channel_other
#         self.mask_channel_length = cfg.mask_channel_length
#         self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
#         self.mask_channel_min_space = cfg.mask_channel_min_space

#         self.dropout_input = nn.Dropout(cfg.dropout_input)
#         self.dropout_features = nn.Dropout(cfg.dropout_features)

#         self.feature_grad_mult = cfg.feature_grad_mult

#         self.quantizer = None
#         self.input_quantizer = None

#         self.n_negatives = cfg.num_negatives
#         self.cross_sample_negatives = cfg.cross_sample_negatives
#         self.codebook_negatives = cfg.codebook_negatives
#         self.negatives_from_everywhere = cfg.negatives_from_everywhere

#         self.logit_temp = cfg.logit_temp

#         final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim

#         if cfg.quantize_targets:
#             vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else final_dim
#             self.quantizer = GumbelVectorQuantizer(
#                 dim=self.embed,
#                 num_vars=cfg.latent_vars,
#                 temp=cfg.latent_temp,
#                 groups=cfg.latent_groups,
#                 combine_groups=False,
#                 vq_dim=vq_dim,
#                 time_first=True,
#             )
#             self.project_q = nn.Linear(vq_dim, final_dim)
#         else:
#             self.project_q = nn.Linear(self.embed, final_dim)

#         if cfg.quantize_input:
#             if cfg.same_quantizer and self.quantizer is not None:
#                 vq_dim = final_dim
#                 self.input_quantizer = self.quantizer
#             else:
#                 vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else cfg.encoder_embed_dim
#                 self.input_quantizer = GumbelVectorQuantizer(
#                     dim=self.embed,
#                     num_vars=cfg.latent_vars,
#                     temp=cfg.latent_temp,
#                     groups=cfg.latent_groups,
#                     combine_groups=False,
#                     vq_dim=vq_dim,
#                     time_first=True,
#                 )
#             self.project_inp = nn.Linear(vq_dim, cfg.encoder_embed_dim)

#         self.mask_emb = nn.Parameter(
#             torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
#         )

#         self.encoder = TransformerEncoder(cfg)
#         self.layer_norm = LayerNorm(self.embed)

#         self.target_glu = None
#         if cfg.target_glu:
#             self.target_glu = nn.Sequential(
#                 nn.Linear(final_dim, final_dim * 2), nn.GLU()
#             )

#         self.final_proj = nn.Linear(cfg.encoder_embed_dim, final_dim)

#     def upgrade_state_dict_named(self, state_dict, name):
#         super().upgrade_state_dict_named(state_dict, name)
#         """Upgrade a (possibly old) state dict for new versions of fairseq."""
#         return state_dict


#     def apply_mask(self, x, padding_mask):
#         B, T, C = x.shape
#         if self.mask_prob > 0:
#             mask_indices = compute_mask_indices(
#                 (B, T),
#                 padding_mask,
#                 self.mask_prob,
#                 self.mask_length,
#                 self.mask_selection,
#                 self.mask_other,
#                 min_masks=2,
#                 no_overlap=self.no_mask_overlap,
#                 min_space=self.mask_min_space,
#             )
#             mask_indices = torch.from_numpy(mask_indices).to(x.device)
#             x[mask_indices] = self.mask_emb
#         else:
#             mask_indices = None

#         if self.mask_channel_prob > 0:
#             mask_channel_indices = compute_mask_indices(
#                 (B, C),
#                 None,
#                 self.mask_channel_prob,
#                 self.mask_channel_length,
#                 self.mask_channel_selection,
#                 self.mask_channel_other,
#                 no_overlap=self.no_mask_channel_overlap,
#                 min_space=self.mask_channel_min_space,
#             )
#             mask_channel_indices = (
#                 torch.from_numpy(mask_channel_indices)
#                 .to(x.device)
#                 .unsqueeze(1)
#                 .expand(-1, T, -1)
#             )
#             x[mask_channel_indices] = 0

#         return x, mask_indices

#     def sample_negatives(self, y, num):

#         if self.n_negatives == 0 and self.cross_sample_negatives == 0:
#             return y.new(0)

#         bsz, tsz, fsz = y.shape
#         y = y.view(-1, fsz)  # BTC => (BxT)C

#         cross_high = tsz * bsz
#         high = tsz
#         with torch.no_grad():
#             assert high > 1, f"{bsz,tsz,fsz}"

#             if self.n_negatives > 0:
#                 tszs = (
#                     buffered_arange(num)
#                     .unsqueeze(-1)
#                     .expand(-1, self.n_negatives)
#                     .flatten()
#                 )

#                 neg_idxs = torch.randint(
#                     low=0, high=high - 1, size=(bsz, self.n_negatives * num)
#                 )
#                 neg_idxs[neg_idxs >= tszs] += 1

#             if self.cross_sample_negatives > 0:
#                 tszs = (
#                     buffered_arange(num)
#                     .unsqueeze(-1)
#                     .expand(-1, self.cross_sample_negatives)
#                     .flatten()
#                 )

#                 cross_neg_idxs = torch.randint(
#                     low=0,
#                     high=cross_high - 1,
#                     size=(bsz, self.cross_sample_negatives * num),
#                 )
#                 cross_neg_idxs[cross_neg_idxs >= tszs] += 1

#         if self.n_negatives > 0:
#             for i in range(1, bsz):
#                 neg_idxs[i] += i * high
#         else:
#             neg_idxs = cross_neg_idxs

#         if self.cross_sample_negatives > 0 and self.n_negatives > 0:
#             neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

#         negs = y[neg_idxs.view(-1)]
#         negs = negs.view(
#             bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
#         ).permute(
#             2, 0, 1, 3
#         )  # to NxBxTxC
#         return negs, neg_idxs

#     def compute_preds(self, x, y, negatives):

#         neg_is_pos = (y == negatives).all(-1)
#         y = y.unsqueeze(0)
#         targets = torch.cat([y, negatives], dim=0)

#         logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

#         logits /= self.logit_temp

#         if neg_is_pos.any():
#             logits[1:][neg_is_pos] = float("-inf")

#         return logits

#     def forward(self, source, padding_mask=None, mask=True, features_only=False):

#         if self.feature_grad_mult > 0:
#             features = self.feature_extractor(source)
#             if self.feature_grad_mult != 1.0:
#                 features = GradMultiply.apply(features, self.feature_grad_mult)
#         else:
#             with torch.no_grad():
#                 features = self.feature_extractor(source)

#         features_pen = features.float().pow(2).mean()

#         features = features.transpose(1, 2)
#         features = self.layer_norm(features)
#         unmasked_features = features.clone()

#         if padding_mask is not None:
#             extra = padding_mask.size(1) % features.size(1)
#             if extra > 0:
#                 padding_mask = padding_mask[:, :-extra]
#             padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
#             padding_mask = padding_mask.all(-1)

#         if self.post_extract_proj is not None:
#             features = self.post_extract_proj(features)

#         features = self.dropout_input(features)
#         unmasked_features = self.dropout_features(unmasked_features)

#         num_vars = None
#         code_ppl = None
#         prob_ppl = None
#         curr_temp = None

#         if self.input_quantizer:
#             q = self.input_quantizer(features, produce_targets=False)
#             features = q["x"]
#             num_vars = q["num_vars"]
#             code_ppl = q["code_perplexity"]
#             prob_ppl = q["prob_perplexity"]
#             curr_temp = q["temp"]
#             features = self.project_inp(features)

#         if mask:
#             x, mask_indices = self.apply_mask(features, padding_mask)
#             if mask_indices is not None:
#                 y = unmasked_features[mask_indices].view(
#                     unmasked_features.size(0), -1, unmasked_features.size(-1)
#                 )
#             else:
#                 y = unmasked_features
#         else:
#             x = features
#             y = unmasked_features
#             mask_indices = None

#         x = self.encoder(x, padding_mask=padding_mask)

#         if features_only:
#             return {"x": x, "padding_mask": padding_mask}

#         if self.quantizer:
#             q = self.quantizer(y, produce_targets=False)
#             y = q["x"]
#             num_vars = q["num_vars"]
#             code_ppl = q["code_perplexity"]
#             prob_ppl = q["prob_perplexity"]
#             curr_temp = q["temp"]

#             y = self.project_q(y)

#             if self.negatives_from_everywhere:
#                 neg_cands, *_ = self.quantizer(unmasked_features, produce_targets=False)
#                 negs, _ = self.sample_negatives(neg_cands, y.size(1))
#                 negs = self.project_q(negs)

#             else:
#                 negs, _ = self.sample_negatives(y, y.size(1))

#             if self.codebook_negatives > 0:
#                 cb_negs = self.quantizer.sample_from_codebook(
#                     y.size(0) * y.size(1), self.codebook_negatives
#                 )
#                 cb_negs = cb_negs.view(
#                     self.codebook_negatives, y.size(0), y.size(1), -1
#                 )  # order doesnt matter
#                 cb_negs = self.project_q(cb_negs)
#                 negs = torch.cat([negs, cb_negs], dim=0)
#         else:
#             y = self.project_q(y)

#             if self.negatives_from_everywhere:
#                 negs, _ = self.sample_negatives(unmasked_features, y.size(1))
#                 negs = self.project_q(negs)
#             else:
#                 negs, _ = self.sample_negatives(y, y.size(1))

#         x = x[mask_indices].view(x.size(0), -1, x.size(-1))

#         if self.target_glu:
#             y = self.target_glu(y)
#             negs = self.target_glu(negs)

#         x = self.final_proj(x)
#         x = self.compute_preds(x, y, negs)

#         result = {"x": x, "padding_mask": padding_mask, "features_pen": features_pen}

#         if prob_ppl is not None:
#             result["prob_perplexity"] = prob_ppl
#             result["code_perplexity"] = code_ppl
#             result["num_vars"] = num_vars
#             result["temp"] = curr_temp

#         return result

#     def quantize(self, x):
#         assert self.quantizer is not None
#         x = self.feature_extractor(x)
#         x = x.transpose(1, 2)
#         x = self.layer_norm(x)
#         return self.quantizer.forward_idx(x)

#     def extract_features(self, source, padding_mask, mask=False):
#         res = self.forward(source, padding_mask, mask=mask, features_only=True)
#         return res["x"], res["padding_mask"]

#     def get_logits(self, net_output):
#         logits = net_output["x"]
#         logits = logits.transpose(0, 2)
#         logits = logits.reshape(-1, logits.size(-1))
#         return logits

#     def get_targets(self, sample, net_output, expand_steps=True):
#         x = net_output["x"]
#         return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

#     def get_extra_losses(self, net_output):
#         pen = []

#         if "prob_perplexity" in net_output:
#             pen.append(
#                 (net_output["num_vars"] - net_output["prob_perplexity"])
#                 / net_output["num_vars"]
#             )

#         if "features_pen" in net_output:
#             pen.append(net_output["features_pen"])

#         return pen

#     def remove_pretraining_modules(self):
#         self.quantizer = None
#         self.project_q = None
#         self.target_glu = None
#         self.final_proj = None


# class ConvFeatureExtractionModel(nn.Module):
#     def __init__(
#         self,
#         conv_layers: List[Tuple[int, int, int]],
#         dropout: float = 0.0,
#         mode: str = "default",
#         conv_bias: bool = False,
#     ):
#         super().__init__()

#         assert mode in {"default", "layer_norm"}

#         def block(
#             n_in,
#             n_out,
#             k,
#             stride,
#             is_layer_norm=False,
#             is_group_norm=False,
#             conv_bias=False,
#         ):
#             def make_conv():
#                 conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
#                 nn.init.kaiming_normal_(conv.weight)
#                 return conv

#             assert (
#                 is_layer_norm and is_group_norm
#             ) == False, "layer norm and group norm are exclusive"

#             if is_layer_norm:
#                 return nn.Sequential(
#                     make_conv(),
#                     nn.Dropout(p=dropout),
#                     nn.Sequential(
#                         TransposeLast(),
#                         Fp32LayerNorm(dim, elementwise_affine=True),
#                         TransposeLast(),
#                     ),
#                     nn.GELU(),
#                 )
#             elif is_group_norm:
#                 return nn.Sequential(
#                     make_conv(),
#                     nn.Dropout(p=dropout),
#                     Fp32GroupNorm(dim, dim, affine=True),
#                     nn.GELU(),
#                 )
#             else:
#                 return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

#         in_d = 1
#         self.conv_layers = nn.ModuleList()
#         for i, cl in enumerate(conv_layers):
#             assert len(cl) == 3, "invalid conv definition: " + str(cl)
#             (dim, k, stride) = cl

#             self.conv_layers.append(
#                 block(
#                     in_d,
#                     dim,
#                     k,
#                     stride,
#                     is_layer_norm=mode == "layer_norm",
#                     is_group_norm=mode == "default" and i == 0,
#                     conv_bias=conv_bias,
#                 )
#             )
#             in_d = dim

#     def forward(self, x):

#         # BxT -> BxCxT
#         x = x.unsqueeze(1)

#         for conv in self.conv_layers:
#             x = conv(x)

#         return x


# class TransformerEncoder(nn.Module):
#     def __init__(self, args):
#         super().__init__()

#         self.dropout = args.dropout
#         self.embedding_dim = args.encoder_embed_dim

#         self.pos_conv = nn.Conv1d(
#             self.embedding_dim,
#             self.embedding_dim,
#             kernel_size=args.conv_pos,
#             padding=args.conv_pos // 2,
#             groups=args.conv_pos_groups,
#         )
#         dropout = 0
#         std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
#         nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
#         nn.init.constant_(self.pos_conv.bias, 0)

#         self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
#         self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

#         self.layers = nn.ModuleList(
#             [
#                 TransformerSentenceEncoderLayer(
#                     embedding_dim=self.embedding_dim,
#                     ffn_embedding_dim=args.encoder_ffn_embed_dim,
#                     num_attention_heads=args.encoder_attention_heads,
#                     dropout=self.dropout,
#                     attention_dropout=args.attention_dropout,
#                     activation_dropout=args.activation_dropout,
#                     activation_fn=args.activation_fn,
#                     layer_norm_first=args.layer_norm_first,
#                 )
#                 for _ in range(args.encoder_layers)
#             ]
#         )

#         self.layer_norm_first = args.layer_norm_first
#         self.layer_norm = LayerNorm(self.embedding_dim)
#         self.layerdrop = args.encoder_layerdrop

#         self.apply(init_bert_params)

#     def forward(self, x, padding_mask=None):
#         x = self.extract_features(x, padding_mask)

#         if self.layer_norm_first:
#             x = self.layer_norm(x)

#         return x

#     def extract_features(self, x, padding_mask=None):

#         if padding_mask is not None:
#             x[padding_mask] = 0

#         x_conv = self.pos_conv(x.transpose(1, 2))
#         x_conv = x_conv.transpose(1, 2)
#         x += x_conv

#         if not self.layer_norm_first:
#             x = self.layer_norm(x)

#         x = F.dropout(x, p=self.dropout, training=self.training)

#         # B x T x C -> T x B x C
#         x = x.transpose(0, 1)

#         layer_results = []
#         for i, layer in enumerate(self.layers):
#             dropout_probability = np.random.random()
#             if not self.training or (dropout_probability > self.layerdrop):
#                 x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
#                 layer_results.append(x)

#         # T x B x C -> B x T x C
#         x = x.transpose(0, 1)

#         return x

#     def max_positions(self):
#         """Maximum output length supported by the encoder."""
#         return self.args.max_positions

#     def upgrade_state_dict_named(self, state_dict, name):
#         """Upgrade a (possibly old) state dict for new versions of fairseq."""
#         return state_dict


# class TransformerSentenceEncoderLayer(nn.Module):
#     """
#     Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
#     models.
#     """

#     def __init__(
#         self,
#         embedding_dim: float = 768,
#         ffn_embedding_dim: float = 3072,
#         num_attention_heads: float = 8,
#         dropout: float = 0.1,
#         attention_dropout: float = 0.1,
#         activation_dropout: float = 0.1,
#         activation_fn: str = "relu",
#         layer_norm_first: bool = False,
#     ) -> None:

#         super().__init__()
#         # Initialize parameters
#         self.embedding_dim = embedding_dim
#         self.dropout = dropout
#         self.activation_dropout = activation_dropout

#         # Initialize blocks
#         self.activation_fn = utils.get_activation_fn(activation_fn)
#         self.self_attn = MultiheadAttention(
#             self.embedding_dim,
#             num_attention_heads,
#             dropout=attention_dropout,
#             self_attention=True,
#         )

#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(self.activation_dropout)
#         self.dropout3 = nn.Dropout(dropout)

#         self.layer_norm_first = layer_norm_first

#         # layer norm associated with the self attention layer
#         self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
#         self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
#         self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

#         # layer norm associated with the position wise feed-forward NN
#         self.final_layer_norm = LayerNorm(self.embedding_dim)

#     def forward(
#         self,
#         x: torch.Tensor,
#         self_attn_mask: torch.Tensor = None,
#         self_attn_padding_mask: torch.Tensor = None,
#         need_weights: bool = False,
#         att_args=None,
#     ):
#         """
#         LayerNorm is applied either before or after the self-attention/ffn
#         modules similar to the original Transformer imlementation.
#         """
#         residual = x

#         if self.layer_norm_first:
#             x = self.self_attn_layer_norm(x)
#             x, attn = self.self_attn(
#                 query=x,
#                 key=x,
#                 value=x,
#                 key_padding_mask=self_attn_padding_mask,
#                 need_weights=False,
#                 attn_mask=self_attn_mask,
#             )
#             x = self.dropout1(x)
#             x = residual + x

#             residual = x
#             x = self.final_layer_norm(x)
#             x = self.activation_fn(self.fc1(x))
#             x = self.dropout2(x)
#             x = self.fc2(x)
#             x = self.dropout3(x)
#             x = residual + x
#         else:
#             x, attn = self.self_attn(
#                 query=x,
#                 key=x,
#                 value=x,
#                 key_padding_mask=self_attn_padding_mask,
#                 need_weights=need_weights,
#             )

#             x = self.dropout1(x)
#             x = residual + x

#             x = self.self_attn_layer_norm(x)

#             residual = x
#             x = self.activation_fn(self.fc1(x))
#             x = self.dropout2(x)
#             x = self.fc2(x)
#             x = self.dropout3(x)
#             x = residual + x
#             x = self.final_layer_norm(x)

#         return x, attn


