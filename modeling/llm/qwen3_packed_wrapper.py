import torch
from torch import nn
from typing import Optional, Union
from flash_attn import flash_attn_varlen_func
from einops import rearrange

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

logger = logging.get_logger(__name__)

class Qwen3PackedAttention(Qwen3Attention):
    """
    Qwen3 Attention module modified to support packed sequences for efficient training
    and inference with variable-length inputs. This module uses flash_attn_varlen_func.
    
    It inherits from Qwen3Attention to reuse the projection layers and configuration.
    """
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None, # This will be ignored in packed mode
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
                "Qwen3PackedAttention received `sample_lens=None`. Falling back to the standard attention mechanism. "
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

        # 4. Prepare for Flash Attention Varlen
        # Create cumulative sequence lengths for flash_attn_varlen_func
        cu_seqlens = nn.functional.pad(torch.cumsum(sample_lens, dim=0, dtype=torch.int32), (1, 0))
        max_seqlen = sample_lens.max().item()

        # 5. Call Flash Attention Varlen Function
        # Note: flash_attn_varlen_func expects inputs of shape [total_tokens, num_heads, head_dim]
        attn_output = flash_attn_varlen_func(
            q=query_states.to(torch.bfloat16),
            k=key_states.to(torch.bfloat16),
            v=value_states.to(torch.bfloat16),
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=True,
            # The scaling factor is applied inside the kernel
            softmax_scale=self.scaling,
        )

        # 6. Final Projection
        # Reshape and apply the output projection
        attn_output = attn_output.reshape(total_tokens, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if attn_output.ndim == 2:
            attn_output = attn_output.unsqueeze(0)

        # Packed attention does not return attention weights or handle past_key_values
        return attn_output, None


class Qwen3ForCausalLMWrapper(Qwen3ForCausalLM):
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
            packed_attn_module = Qwen3PackedAttention(config=self.config, layer_idx=layer.self_attn.layer_idx)
            packed_attn_module.load_state_dict(layer.self_attn.state_dict())
            layer.self_attn = packed_attn_module

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, # Will be ignored if sample_lens is provided
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
            
            attention_mask = None # Not used in varlen attention
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



# Below is for test and debug
from transformers import AutoTokenizer

def build_model_and_tokenizer(model_name: str, device: str):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model with packed attention wrapper...")
    config = Qwen3Config.from_pretrained(model_name)
    model = Qwen3ForCausalLMWrapper.from_pretrained(
        model_name,
        config=config,
        use_packed_attn=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    return model, tokenizer


def prepare_prompts(tokenizer, prompts, device):
    # Tokenize each prompt individually -> list of [1, seq_len_i] tensors
    tokenized_prompts = [tokenizer.encode(p, return_tensors="pt").to(device) for p in prompts]

    # Packed input_ids: [1, total_tokens]
    packed_input_ids = torch.cat(tokenized_prompts, dim=1).to(device)

    # sample_lens: [num_sequences], lengths of each original sequence
    sample_lens = torch.tensor([t.shape[1] for t in tokenized_prompts], dtype=torch.long, device=device)

    # position_ids for packed input
    packed_position_ids = torch.arange(0, packed_input_ids.shape[1], device=device).unsqueeze(0)

    return tokenized_prompts, packed_input_ids, packed_position_ids, sample_lens


def test_kv_cache(model, tokenizer, tokenized_prompts, max_new_tokens: int = 20):
    print("\n--- KV Cache Generation ---")
    generated_outputs = []

    for prompt_tokens in tokenized_prompts:
        input_ids = prompt_tokens
        past_key_values = None
        generated_ids = [input_ids]

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            generated_ids.append(next_token_id)
            input_ids = next_token_id
            past_key_values = outputs.past_key_values

        generated_outputs.append(torch.cat(generated_ids, dim=1))

    # Decode
    print(f"\n--- Decoded Generations (max_new_tokens={max_new_tokens}) ---")
    return [tokenizer.decode(seq.squeeze(0), skip_special_tokens=True) for seq in generated_outputs]


def test_packed_single_pass(
    model,
    tokenizer,
    packed_input_ids,
    packed_position_ids,
    sample_lens,
    max_new_tokens: int = 1,  # default single-step
):
    print("\n--- Packed Attention: Multi-Step (repacked each step) ---")

    # Initialize sequences per prompt as lists of tensors [1, seq_len_i]
    # Start from the original split sequences reconstructed from the packed inputs
    # Note: We can reconstruct by slicing using sample_lens.
    sequences = []
    start = 0
    for L in sample_lens.tolist():
        sequences.append(packed_input_ids[:, start:start + L].clone())
        start += L

    predicted_next_token_ids_all_steps = []  # list of per-step next token ids

    for step in range(max_new_tokens):
        # Re-pack current sequences at each step
        packed_input_ids_step = torch.cat(sequences, dim=1)
        packed_position_ids_step = torch.arange(
            0, packed_input_ids_step.shape[1], device=packed_input_ids_step.device
        ).unsqueeze(0)

        with torch.no_grad():
            outputs = model(
                input_ids=packed_input_ids_step,
                position_ids=packed_position_ids_step,
                sample_lens=sample_lens,  # triggers packed attention path
            )
            logits = outputs.logits

        # Next token prediction per sequence: take logits at each sequence's last token
        end_indices = torch.cumsum(sample_lens, dim=0) - 1
        last_token_logits = logits[0, end_indices, :]
        predicted_next_token_ids = torch.argmax(last_token_logits, dim=-1)  # [num_sequences]

        # Record this step's predictions
        predicted_next_token_ids_all_steps.append(predicted_next_token_ids.tolist())

        # Append new tokens to each sequence (shape [1, 1] per sequence)
        for i in range(len(sequences)):
            new_token = predicted_next_token_ids[i].view(1, 1)
            sequences[i] = torch.cat([sequences[i], new_token], dim=1)

        # Important: update sample_lens to reflect growth
        sample_lens = sample_lens + 1  # each sequence grew by 1 token

    return predicted_next_token_ids_all_steps


def print_kv_cache_results_one_line(prompts, kv_texts):
    # Format: [idx] Prompt: "..."; Generated: "..."
    for i, txt in enumerate(kv_texts):
        print(f'[{i}] Prompt: "{prompts[i]}"; Generated: "{txt}"')


def print_packed_multistep_results_one_line(prompts, pred_ids_steps, tokenizer, join_sep=" "):
    """
    pred_ids_steps: List[List[int]] with shape [num_steps][num_sequences]
    Output format (per line): [idx] Prompt: "..."; Generated: "..."
    The Generated field is all decoded next tokens across steps, joined by join_sep.
    """
    num_sequences = len(prompts)

    # Transpose steps -> per-sequence token IDs
    per_seq_token_ids = [[] for _ in range(num_sequences)]
    for step_ids in pred_ids_steps:
        for i, tid in enumerate(step_ids):
            per_seq_token_ids[i].append(tid)

    # Print one line per sequence, matching KV-cache format
    for i, prompt in enumerate(prompts):
        decoded_tokens = [tokenizer.decode(tid) for tid in per_seq_token_ids[i]]
        generated_str = join_sep.join(decoded_tokens).replace("\n", "\\n")
        print(f'[{i}] Prompt: "{prompt}"; Generated: "{generated_str}"')


def main():
    # --- 1. Init ---
    model_name = "Qwen/Qwen3-0.6B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = build_model_and_tokenizer(model_name, device)

    # --- 2. Data ---
    prompts = [
        "The capital of France is",
        "The highest mountain in the world is Mount",
        "To be, or not to be, that is the question:",
    ]
    tokenized_prompts, packed_input_ids, packed_position_ids, sample_lens = prepare_prompts(tokenizer, prompts, device)

    # --- 3. KV Cache Test ---
    kv_texts = test_kv_cache(model, tokenizer, tokenized_prompts, max_new_tokens=20)
    print_kv_cache_results_one_line(prompts, kv_texts)

    # --- 4. Packed Single-Pass Test ---
    pred_ids_steps = test_packed_single_pass(model, tokenizer, packed_input_ids, packed_position_ids, sample_lens, max_new_tokens=20)
    print_packed_multistep_results_one_line(prompts, pred_ids_steps, tokenizer, join_sep=" ")

if __name__ == "__main__":
    main()


