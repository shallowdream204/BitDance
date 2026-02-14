import torch
from typing import Optional, Union

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    apply_rotary_pos_emb,
    CausalLMOutputWithPast,
    Qwen3Config
)
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging

from torch.nn.attention.flex_attention import flex_attention

torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096

flex_attention = torch.compile(flex_attention)

logger = logging.get_logger(__name__)

def pad_sequence(tensor, pad_size):
    H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=1)

class PackedFlexAttention(Qwen3Attention):
    """
    Qwen3 Attention module modified to support packed sequences for efficient training
    and inference with variable-length inputs. This module uses flex_attention.
    
    It inherits from Qwen3Attention to reuse the projection layers and configuration.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        sample_lens: Optional[torch.LongTensor] = None, # New argument for packed sequences
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # If sample_lens is not provided, it means we are not using packed attention.
        # Fall back to the original Qwen3Attention's forward method.
        # This makes the module versatile for both packed and padded inputs.
        if sample_lens is None:
            logger.warning_once(
                "PackedFlexAttention received `sample_lens=None`. Falling back to the standard attention mechanism. "
                "This may be slow. For packed attention, ensure `sample_lens` is provided."
            )
            # Call the parent class's forward method
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs
            )

        # --- Packed Attention Logic ---
        
        # The input hidden_states is already a packed tensor of shape [total_tokens, hidden_size]
        total_tokens = hidden_states.shape[-2]

        # 1. Project Q, K, V from the packed hidden_states
        # Input: [total_tokens, hidden_size] -> Output: [total_tokens, num_heads, head_dim]
        query_states = self.q_norm(self.q_proj(hidden_states).view(total_tokens, self.config.num_attention_heads, self.head_dim))
        key_states = self.k_norm(self.k_proj(hidden_states).view(total_tokens, self.config.num_key_value_heads, self.head_dim))
        value_states = self.v_proj(hidden_states).view(total_tokens, self.config.num_key_value_heads, self.head_dim)

        # 2. Apply Rotary Positional Embeddings (RoPE)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos.squeeze(0), sin.squeeze(0))

        # 3. Pad sequences to the max length in the batch
        pad_size = sum(sample_lens) - query_states.shape[0]
        query_states = pad_sequence(query_states.permute(1, 0, 2), pad_size)
        key_states = pad_sequence(key_states.permute(1, 0, 2), pad_size)
        value_states = pad_sequence(value_states.permute(1, 0, 2), pad_size)

        # 4. Apply flex_attention
        attn_output = flex_attention(
                query_states.to(torch.bfloat16).unsqueeze(0), 
                key_states.to(torch.bfloat16).unsqueeze(0), 
                value_states.to(torch.bfloat16).unsqueeze(0), 
                enable_gqa=True,
                block_mask=attention_mask,
            )
        end_index = attn_output.shape[2] - pad_size
        attn_output = attn_output[0, :, :end_index, :] # H, L, D

        # 5. Final Projection
        # Reshape and apply the output projection
        attn_output = attn_output.transpose(0, 1).reshape(total_tokens, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if attn_output.ndim == 2:
            attn_output = attn_output.unsqueeze(0)

        # Packed attention does not return attention weights or handle past_key_values
        return attn_output, None


class Qwen3FlexWrapper(Qwen3ForCausalLM):
    """
    A wrapper for Qwen3ForCausalLM that enables packed attention by monkey-patching
    the attention modules.
    """
    def __init__(self, config: Qwen3Config, use_packed_attn: bool = True, **kwargs):
        # Initialize the original model first
        super().__init__(config, **kwargs)

        self.use_packed_attn = use_packed_attn
        if self.use_packed_attn:
            self._enable_packed_attention()

    def _enable_packed_attention(self):
        """
        Replaces the standard Qwen3Attention modules with our Qwen3PackedAttention module.
        """
        logger.info("Enabling packed attention by replacing Qwen3Attention with Qwen3PackedAttention.")
        for layer in self.model.layers:
            packed_attn_module = PackedFlexAttention(config=self.config, layer_idx=layer.self_attn.layer_idx)
            packed_attn_module.load_state_dict(layer.self_attn.state_dict())
            layer.self_attn = packed_attn_module

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        sample_lens: Optional[torch.LongTensor] = None, # New argument
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:

        is_packed_path = self.use_packed_attn and sample_lens is not None
        if is_packed_path:
            if input_ids is not None and input_ids.shape[0] > 1:
                raise ValueError("For packed attention, `input_ids` must be a packed tensor with batch_size=1.")
            
            # attention_mask = None # Not used in varlen attention
            kwargs["sample_lens"] = sample_lens

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )