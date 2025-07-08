import torch
import torch.nn as nn
from typing import Optional, Dict, Union
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
)
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.parallel.pipeline_parallel.block import TensorPointer
from nanotron import distributed as dist


class MTPLayer(nn.Module):
    def __init__(
        self,
        config,
        token_embedding,
        layer_idx: int,
        mtp_layer_id: int,
        parallel_config,
        tp_pg: dist.ProcessGroup,
    ):
        super().__init__()
        
        self.layer_idx = layer_idx
        self.mtp_layer_id = mtp_layer_id
        self.recompute_layer = parallel_config.recompute_layer
        
        self.norm_prev_hidden = TritonRMSNorm(
            config.hidden_size, 
            eps=config.rms_norm_eps
        )
        self.norm_curr_embd = TritonRMSNorm(
            config.hidden_size, 
            eps=config.rms_norm_eps
        )
        
        self.projection = TensorParallelColumnLinear(
            2 * config.hidden_size,
            config.hidden_size,
            pg=tp_pg,
            bias=False,
            gather_output=False,
        )
        
        # 核心Transformer块
        from nanotron.models.llama import LlamaDecoderLayer
        self.transformer_block = LlamaDecoderLayer(
            config=config,
            parallel_config=parallel_config,
            tp_pg=tp_pg,
            layer_idx=layer_idx,
        )
        
        # 共享token嵌入
        self.token_embedding = token_embedding

    def _core_forward(
        self,
        prev_hidden_states: torch.Tensor,
        input_embed: torch.Tensor,
        sequence_mask: torch.Tensor,
    ):
        """MTP层的核心前向传播逻辑"""
        
        # prev_hidden_states: [pre_seq_len, batch, hidden] -> [pre_seq_len-1, batch, hidden]
        prev_hidden_states = prev_hidden_states[:-1, :, :]
        prev_normalized = self.norm_prev_hidden(prev_hidden_states)

        # input_embed: [seq_len, batch, hidden] -> [seq_len-mtp_layer_id, batch, hidden]
        input_embed = input_embed[self.mtp_layer_id:, :, :]
        curr_normalized = self.norm_curr_embd(input_embed)

        # adjusted_mask是跟prev_hidden_states对齐
        adjusted_mask = sequence_mask[:, :-self.mtp_layer_id] if self.mtp_layer_id > 0 else sequence_mask[:, :-1]    

        fused_features = torch.cat([prev_normalized, curr_normalized], dim=-1)
        hidden_states = self.projection(fused_features)
        output = self.transformer_block(
            hidden_states=hidden_states,
            sequence_mask=adjusted_mask,
        )
        
        return output["hidden_states"]

    def forward(
        self,
        prev_hidden_states: Union[torch.Tensor, TensorPointer],
        input_embed: Union[torch.Tensor, TensorPointer],
        sequence_mask: Union[torch.Tensor, TensorPointer],
    ):
        if self.recompute_layer and not isinstance(prev_hidden_states, TensorPointer):
            # 使用梯度重计算
            from torch.utils.checkpoint import CheckpointFunction
            return CheckpointFunction.apply(
                self._core_forward,
                True,
                prev_hidden_states,
                input_embed,
                sequence_mask,
            )
        else:
            return self._core_forward(prev_hidden_states, input_embed, sequence_mask)