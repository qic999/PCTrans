# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import io
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from .attention import MultiheadAttention
import math
from torch.cuda.amp import autocast
from detectron2.layers.batch_norm import get_norm

def gen_sineembed_for_position(pos_tensor, temperature=20):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / 128)
    pos = []
    # import pdb; pdb.set_trace()
    for i in range(int(pos_tensor.shape[-1]//2)):
        x_embed = pos_tensor[:, :, 2*i] * scale
        y_embed = pos_tensor[:, :, 2*i+1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos.append(pos_y)
        pos.append(pos_x)
    pos = torch.cat(pos, dim=2)
    return pos

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256

        q_content = self.sa_qcontent_proj(tgt)     # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        # num_queries, bs, n_model = q_content.shape
        # hw, _, _ = k_content.shape

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False,points_num=1):
        super().__init__()
        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model*2*points_num, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False):
        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos) # [950, 2, 256]

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        keep_query_pos = False
        if is_first or keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        if query_sine_embed is not None:
            # import pdb
            # pdb.set_trace()
            query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed) 
            query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead) # [300, 2, 8, 32]
            q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        else:
            q_pos = q_pos.view(num_queries, bs, self.nhead, n_model//self.nhead)
            q = torch.cat([q, q_pos], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                                   key=k,
                                   value=v, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]               
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed = None,
                is_first = False):
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos, query_sine_embed, is_first)

class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

@TRANSFORMER_DECODER_REGISTRY.register()
class MultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        points_num,
        sem_loss_on,
        norm,
        rel_coord
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        # assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    points_num=points_num
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.hidden_dim = hidden_dim
        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        # self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # position query
        self.ref_point_head = MLP(hidden_dim, hidden_dim, points_num*2, 2)
        for contr in self.ref_point_head.layers:
            nn.init.xavier_uniform_(contr.weight)
            nn.init.zeros_(contr.bias)

        self.query_scale = MLP(hidden_dim, hidden_dim*2, hidden_dim*2*points_num, 2)
        for contr in self.query_scale.layers:
            nn.init.xavier_uniform_(contr.weight)
            nn.init.zeros_(contr.bias)


        # point update
        self.point_embed_diff_each_layer = False
        if self.point_embed_diff_each_layer:
            self.point_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 2*points_num, 3) for i in range(self.num_layers)])
        else:
            self.point_embed = MLP(hidden_dim, hidden_dim, 2*points_num, 3)
        for contr in self.point_embed.layers:
            nn.init.xavier_uniform_(contr.weight)
            nn.init.zeros_(contr.bias)

        # self.in_channels = hidden_dim // 32 # default 8
        self.in_channels = mask_dim
        self.dynamic_mask_channels = 8
        self.controller_layers = 3
        self.mask_out_stride = 4
        self.rel_coord = rel_coord

        # dynamic_mask_head params
        weight_nums, bias_nums = [], []
        for l in range(self.controller_layers):
            if l == 0:
                if self.rel_coord:
                    weight_nums.append((self.in_channels + 2) * self.dynamic_mask_channels)
                else:
                    weight_nums.append(self.in_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)
            elif l == self.controller_layers - 1:
                weight_nums.append(self.dynamic_mask_channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.dynamic_mask_channels * self.dynamic_mask_channels)
                bias_nums.append(self.dynamic_mask_channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.controller = MLP(hidden_dim, hidden_dim, self.num_gen_params, 3)
        for contr in self.controller.layers:
            nn.init.xavier_uniform_(contr.weight)
            nn.init.zeros_(contr.bias)
        
        self.mask_head = torch.nn.Conv2d(hidden_dim, mask_dim, 1, padding=0)
        nn.init.kaiming_uniform_(self.mask_head.weight, a=1)
        nn.init.constant_(self.mask_head.bias, 0)
        # for name, m in self.mask_head.named_modules():
        #     nn.init.kaiming_uniform_(m.weight, a=1)
        #     nn.init.constant_(m.bias, 0)

        self.sem_loss_on = sem_loss_on
        if self.sem_loss_on:
            from .conv_with_kaiming_uniform import conv_with_kaiming_uniform
            conv_block = conv_with_kaiming_uniform(norm, activation=True) # SyncBN
            self.seg_head = nn.Sequential(
                conv_block(hidden_dim, hidden_dim, kernel_size=3, stride=1),
                conv_block(hidden_dim, hidden_dim, kernel_size=3, stride=1)
            )
            self.logits = nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1)

            prior_prob = 0.01 # cfg.MODEL.FCOS.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            torch.nn.init.constant_(self.logits.bias, bias_value)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        # ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret['points_num'] = cfg.MODEL.MASK_FORMER.POSITION_POINTS_NUM

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        
        ret['sem_loss_on'] = cfg.MODEL.MASK_FORMER.SEMANTIC_LOSS_ON
        ret['norm'] = cfg.MODEL.MASK_FORMER.SEMANTIC_NORM
        ret['rel_coord'] = cfg.MODEL.MASK_FORMER.REL_COORD
        return ret

    def forward(self, x, targets, mask_features, mask = None, attn_mask_threshold=0.5, criterion=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_mask = []
        outputs_coords = []
        indices_list = []
        reference_points_before_sigmoid = self.ref_point_head(query_embed) 
        reference_points = reference_points_before_sigmoid.sigmoid()
        ref_points = [reference_points]

        # prediction heads on learnable query features
        # mask_features = mask_features+self.pe_layer(mask_features)
        # outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0], attn_mask_threshold=attn_mask_threshold, targets=targets)            
        if self.sem_loss_on:
            sem_logits_pred = self.logits(self.seg_head(mask_features))
        mask_feat = self.mask_head(mask_features)

        # with autocast(enabled=False):
        dynamic_mask_head_params = self.controller(output)
        outputs_mask, attn_mask = self.dynamic_mask_with_coords(mask_feat, reference_points, dynamic_mask_head_params, mask_feat_stride=4, rel_coord=self.rel_coord, attn_mask_target_size=size_list[0])
        predictions_mask.append(outputs_mask)
        # print('outputs_mask', outputs_mask.shape)
        if not targets == None:
            output_dict = {"pred_masks":outputs_mask}
            indices = criterion.matcher(output_dict, targets)
            indices_list.append(indices)

        for i in range(self.num_layers):
            obj_center = reference_points
            if i == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(output)
            query_sine_embed = gen_sineembed_for_position(obj_center)
            query_sine_embed = query_sine_embed * pos_transformation

            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed,
                query_sine_embed=query_sine_embed, is_first=(i == 0)
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            
            # iter update points
            if self.point_embed is not None:
                if self.point_embed_diff_each_layer:
                    tmp = self.point_embed[i](output)
                else:
                    tmp = self.point_embed(output) # [100, 8, 2]
                # import pdb; pdb.set_trace()
                tmp += inverse_sigmoid(reference_points)
                new_reference_points = tmp.sigmoid()
                if i != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            # with autocast(enabled=False):
            dynamic_mask_head_params = self.controller(output)
            outputs_mask, attn_mask = self.dynamic_mask_with_coords(mask_feat, new_reference_points, dynamic_mask_head_params, mask_feat_stride=4, rel_coord=self.rel_coord, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            # import pdb; pdb.set_trace()
            # print('layer_id', i+1)

            if not targets == None:
                output_dict = {"pred_masks":outputs_mask}
                indices = criterion.matcher(output_dict, targets)
                indices_list.append(indices)
                
            decoder_output = self.decoder_norm(output)
            decoder_output = decoder_output.transpose(0, 1)
            if not self.point_embed_diff_each_layer:
                reference_before_sigmoid = inverse_sigmoid(ref_points[i].transpose(0,1))
                tmp = self.point_embed(decoder_output)
                tmp += reference_before_sigmoid
                outputs_coord = tmp.sigmoid()


            # outputs_mask, attn_mask, outputs_coord = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels], attn_mask_threshold=attn_mask_threshold, targets=targets, layer_id=i, ref_points=ref_points)
            predictions_mask.append(outputs_mask)
            outputs_coords.append(outputs_coord)
            if i == self.num_layers-1:
                if not targets==None:
                    emb_dist = torch.stack([torch.cosine_similarity(output.permute(1,0,2)[i].unsqueeze(1), output.permute(1,0,2)[i].unsqueeze(0), dim=-1) for i in range(bs)])
                    contrast_items_query = select_pos_neg_query(output, emb_dist, indices)
                    contrast_items_mask = select_pos_neg_mask(outputs_mask, emb_dist, indices)
        
        # import matplotlib.pyplot as plt
        # emb_dist_map = emb_dist[0].cpu().numpy()
        # emb_dist_image = plt.imshow(emb_dist_map, cmap='viridis')
        # plt.colorbar(emb_dist_image, fraction=0.046, pad=0.04)
        # plt.savefig('emb_dist_image.png')
        # import pdb; pdb.set_trace()

        outputs_coords = torch.stack(outputs_coords)
        out = {
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(predictions_mask),
            'reference_points':outputs_coords[-1],
            'aux_reference_points': self._set_refpoints_aux_loss(outputs_coords),
            'indices_list':indices_list
        }
        if not targets==None:
            out['pred_qd_query']=contrast_items_query
            # out['pred_qd_region']=contrast_items_region
            out['pred_qd_mask']=contrast_items_mask
        if self.sem_loss_on:
            out['sem_mask'] = sem_logits_pred
        return out
    
    def dynamic_mask_with_coords(self,mask_feats, reference_points, mask_head_params, mask_feat_stride, rel_coord, attn_mask_target_size):
        # print('mask_feats',mask_feats.shape)
        N, in_channels, H, W = mask_feats.size()
        query_num = reference_points.shape[0]
        reference_points = reference_points.transpose(0, 1)
        mask_head_params = mask_head_params.transpose(0, 1)

        orig_h = torch.as_tensor(H*mask_feat_stride).to(reference_points)
        orig_w = torch.as_tensor(W*mask_feat_stride).to(reference_points)
        scale_f = torch.stack([orig_w, orig_h], dim=0)
        reference_points = reference_points * scale_f[None, :] 

        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        # [12544, 2]
        im_inds = torch.cat([torch.tensor(i).repeat(query_num) for i in range(N)],dim=0).tolist()

        if rel_coord:
            instance_locations = reference_points
            relative_coords = instance_locations.reshape(N, query_num, 1, 1, 2) - locations.reshape(1, 1, H, W, 2)
            relative_coords = relative_coords.float()
            relative_coords = relative_coords.permute(0, 1, 4, 2, 3).flatten(-2, -1)
            # import pdb; pdb.set_trace()
            mask_head_inputs = torch.cat([relative_coords, mask_feats[im_inds].reshape(N, query_num, in_channels, H * W)], dim=2) # [4, 100, 10, 12544]
        else:
            mask_head_inputs=mask_feats[im_inds].reshape(N, query_num, in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W) # [1, 4000, 112, 112]

        # mask_head_params: [num_insts_all, num_params]
        mask_head_params = torch.flatten(mask_head_params, 0, 1) # [400, 169]

        weights, biases = parse_dynamic_params(
                mask_head_params, self.dynamic_mask_channels,
                self.weight_nums, self.bias_nums
            )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, mask_head_params.shape[0])
        mask_logits = mask_logits.reshape(N, query_num, H, W)

        attn_mask = F.interpolate(mask_logits, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        resize_h = mask_logits.shape[-2]*2
        resize_w = mask_logits.shape[-1]*2
        mask_logits = F.interpolate(mask_logits, size=(resize_h,resize_w), mode="bilinear", align_corners=False)
        # print('mask_logits',mask_logits.shape)
        return mask_logits, attn_mask

    def mask_heads_forward(self, features, weights, biases, num_insts, FACTOR=1e4):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )

            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size, attn_mask_threshold=0.5, targets=None, layer_id=None, ref_points=None):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        if (not targets == None) or (layer_id==self.num_layers-1):
            resize_h = outputs_mask.shape[-2]*2
            resize_w = outputs_mask.shape[-1]*2
            outputs_mask = F.interpolate(outputs_mask, size=(resize_h,resize_w), mode="bilinear", align_corners=False)

        # import pdb;pdb.set_trace()
        # import imageio as io
        # import numpy as np
        # io.volsave('query_logits.tif', outputs_mask[0].detach().cpu().numpy())
        # io.volsave('query_pred.tif', (outputs_mask[0].detach().cpu().sigmoid()>0.5).numpy().astype(np.uint8)*255)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        # import pdb; pdb.set_trace()
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < attn_mask_threshold).bool()
        attn_mask = attn_mask.detach()

        if layer_id is not None:
            if not self.point_embed_diff_each_layer:
                reference_before_sigmoid = inverse_sigmoid(ref_points[layer_id].transpose(0,1))
                tmp = self.point_embed(decoder_output)
                tmp += reference_before_sigmoid
                outputs_coord = tmp.sigmoid()
            return outputs_mask, attn_mask, outputs_coord

        return outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

    @torch.jit.unused
    def _set_refpoints_aux_loss(self, outputs_coords):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"reference_points": b} for b in outputs_coords[:-1]]


def clutering_based_cos_dist(cos_dist, thre=0.7):
    clutering = []
    rest_idx = set(range(cos_dist.shape[0]))
    for i in range(cos_dist.shape[0]):
        if i not in rest_idx:
            continue
        idx = ((cos_dist[i][i:]>thre).nonzero().squeeze() + i).tolist()
        if not isinstance(idx, list):
            idx = [idx]
        rest_idx = rest_idx-set(idx)
        clutering.append(idx)
    return clutering

def merge_query(query, emb_dist):
    merge_query_emb_list = []
    for i in range(query.shape[0]):
        clutering_idx = clutering_based_cos_dist(emb_dist[i])
        clutering_num = len(clutering_idx)
        print('clutering_idx',clutering_idx)
        print('clutering_num',clutering_num)
        import pdb; pdb.set_trace()
        merge_query_emb = torch.zeros(clutering_num, query.shape[2])
        for j in range(clutering_num):
            merge_query_emb[j] = torch.mean(query[i][clutering_idx[j]], dim=0)
        merge_query_emb_list.append(merge_query_emb)
    return merge_query_emb_list

import random
import copy
def select_pos_neg_query(query, emb_dist, pos_indices, embed_head=None):
    # ref_embeds = embed_head(query) # [2, 300, 256]
    # key_embedds = embed_head(query) # [2, 300, 256]
    query = query.transpose(0, 1)
    query_num = query.shape[1]
    bz = query.shape[0]
    one = torch.tensor(1).to(query) 
    zero = torch.tensor(0).to(query)
    contrast_items = []

    # l2_items = []
    for bz_i in range(bz):
        # clutering_idx = clutering_based_cos_dist(emb_dist[bz_i])
        clutering_idx=[]
        pos_ids = pos_indices[bz_i][0].tolist()
        rest_ids = list(set(range(query_num))-set(pos_ids))
        rest_emb_dist = emb_dist[bz_i][rest_ids][:,pos_ids]
        min_dist_pos = torch.argmax(rest_emb_dist, dim=1).tolist()
        min_dist_pos = torch.tensor([pos_ids[i] for i in min_dist_pos])
        for pos_id in pos_ids:
            clutering= [rest_ids[i] for i in torch.where(min_dist_pos==pos_id)[0].tolist()]
            clutering_idx.append(clutering)  # min_dist_pos==pos_ids[0]

        # import pdb; pdb.set_trace()
        
        for inst_i, (pos_id, clutering_ids) in enumerate(zip(pos_ids,clutering_idx)):
            if len(clutering_ids)==0:
                continue
            # import pdb;pdb.set_trace()
            key_embed_i = query[bz_i][pos_id].unsqueeze(0)
            
            pos_ids = copy.deepcopy(clutering_ids)
            clutering_ids.append(pos_id)
            neg_ids = list(set(range(0,query_num))-set(clutering_ids))
            pos_embed = query[bz_i][pos_ids] # [9, 256]
            neg_embed = query[bz_i][neg_ids] # [254, 256]
            contrastive_embed = torch.cat([pos_embed,neg_embed],dim=0) # [263, 256]
            contrastive_label = torch.cat([one.repeat(len(pos_embed)),zero.repeat(len(neg_embed))],dim=0) # [263]

            contrast = torch.einsum('nc,kc->nk',[contrastive_embed,key_embed_i]) # [263, 1]

            if len(pos_embed) ==0 :
                num_sample_neg = 10
            elif len(pos_embed)*10 >= len(neg_embed):
                num_sample_neg = len(neg_embed)
            else:
                num_sample_neg = len(pos_embed)*10 

            sample_ids = random.sample(list(range(0, len(neg_embed))), num_sample_neg) # len=90

            aux_contrastive_embed = torch.cat([pos_embed,neg_embed[sample_ids]],dim=0) # [99, 256]
            aux_contrastive_label = torch.cat([one.repeat(len(pos_embed)),zero.repeat(num_sample_neg)],dim=0) # [99]
            aux_contrastive_embed=nn.functional.normalize(aux_contrastive_embed.float(),dim=1) #为什么这个需要norm而上面不需要
            key_embed_i=nn.functional.normalize(key_embed_i.float(),dim=1)    
            cosine = torch.einsum('nc,kc->nk',[aux_contrastive_embed,key_embed_i]) # [99, 1]


            contrast_items.append({'contrast':contrast,'label':contrastive_label, 'aux_consin':cosine,'aux_label':aux_contrastive_label})

    return contrast_items


def select_pos_neg_mask(query_mask, emb_dist, pos_indices):
    # ref_embeds = embed_head(query) # [2, 300, 256]
    # key_embedds = embed_head(query) # [2, 300, 256]
    query_num = query_mask.shape[1]
    bz = query_mask.shape[0]
    one = torch.tensor(1).to(query_mask) 
    zero = torch.tensor(0).to(query_mask)
    contrast_items = []

    # import pdb; pdb.set_trace()
    for bz_i in range(bz):
        clutering_idx=[]
        pos_ids = pos_indices[bz_i][0].tolist()
        rest_ids = list(set(range(query_num))-set(pos_ids))
        rest_emb_dist = emb_dist[bz_i][rest_ids][:,pos_ids]
        min_dist_pos = torch.argmax(rest_emb_dist, dim=1).tolist()
        min_dist_pos = torch.tensor([pos_ids[i] for i in min_dist_pos])
        for pos_id in pos_ids:
            clutering= [rest_ids[i] for i in torch.where(min_dist_pos==pos_id)[0].tolist()]
            clutering_idx.append(clutering)  # min_dist_pos==pos_ids[0]
            
        dice_query = dice_for(query_mask[bz_i])
        # import pdb; pdb.set_trace()
        
        for inst_i, (pos_id, clutering_ids) in enumerate(zip(pos_ids,clutering_idx)):
            if len(clutering_ids)==0:
                continue

            pos_ids = copy.deepcopy(clutering_ids)
            clutering_ids.append(pos_id)
            neg_ids = list(set(range(0,query_num))-set(clutering_ids))

            contrastive_label = torch.cat([one.repeat(len(pos_ids)),zero.repeat(len(neg_ids))],dim=0)
            contrast = torch.cat([dice_query[pos_id][pos_ids][:,None], dice_query[pos_id][neg_ids][:,None]])
            # import pdb; pdb.set_trace()
            
            contrast_items.append({'contrast':contrast,'label':contrastive_label})

    return contrast_items

# def dice_for(inputs_1, inputs_2):
#     inputs_1 = inputs_1.flatten(1)
#     inputs_2 = inputs_2.flatten(1)
#     inputs_1 = inputs_1.sigmoid()
#     inputs_2 = inputs_2.sigmoid()
    
#     numerator = inputs_1 @ inputs_2.transpose(-2, -1)
#     inputs_2 = inputs_2.sum(-1)
#     inputs_1 = inputs_1.sum(-1)
#     inputs_1 = inputs_1.view(inputs_1.size(0), 1)
#     inputs_2 = inputs_2.view(1, inputs_2.size(0))
#     denominator = inputs_1 + inputs_2
#     res = (2 * numerator + 1) / (denominator + 1)
#     return res

def dice_for(inputs):
    inputs = inputs.flatten(1)
    inputs = inputs.sigmoid()
    
    numerator = inputs @ inputs.transpose(-2, -1)
    inputs = inputs.sum(-1)
    inputs_1 = inputs.view(inputs.size(0), 1)
    inputs_2 = inputs.view(1, inputs.size(0))
    denominator = inputs_1 + inputs_2
    res = (2 * numerator + 1) / (denominator + 1)
    return res

def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations

def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    # assert len(weight_nums) == len(bias_nums)
    # assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    if len(bias_nums)==0:
        params_splits = list(torch.split_with_sizes(params, weight_nums, dim=1))
        weight_splits = params_splits[:num_layers]

        for l in range(num_layers):
            if l < num_layers - 1:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            else:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
        bias_splits=[]
    else:
        params_splits = list(torch.split_with_sizes(params, weight_nums + bias_nums, dim=1))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                # out_channels x in_channels x 1 x 1
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits