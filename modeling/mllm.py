from typing import List
from tqdm import tqdm
from einops import rearrange
import torch
import torch.nn.functional as F
from torch import nn

from data.data_utils import set_special_token_aliases, add_resolution_special_tokens, add_query_special_tokens

from utils.fs import download
from transformers import (
    PreTrainedModel,  
    AutoTokenizer,
)
from .utils import sample_codebook, flip_tensor_elements_uniform_prob, gaussian_sample, remove_first_user_block, create_sparse_mask, MLPconnector

from torch.nn.attention.flex_attention import create_block_mask


class MLLModel(PreTrainedModel):
    _tied_weights_keys = ["llm_model.lm_head.weight", "vision_head.weight"]   

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.build_llm_model(config)
        self.build_vision_encoder(config)
        self.build_vision_head(config)
        self.build_pos_embed(config)

    def build_pos_embed(self, config):
        max_len = config.head.get("pe_max_len", 4096)
        print(f"build pos embed for max resolution: {max_len}x{max_len}")
        max_len = max_len // config.vit_patch_size
        pos_embed_1d = self._get_1d_sincos_pos_embed(self.hidden_size//2, max_len)
        self.register_buffer(f"pos_embed_1d", pos_embed_1d, persistent=False)

    def _get_1d_sincos_pos_embed(self, dim, max_len, pe_interpolation=1.0):
        assert dim % 2 == 0
        omega = torch.arange(dim // 2, dtype=torch.float32)
        omega /= dim / 2.0
        omega = 1.0 / 10000**omega  # (D/4,)

        pos = torch.arange(max_len, dtype=torch.float32) / pe_interpolation
        out = torch.einsum("m,d->md", pos, omega)  # (max_len, D/4)

        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        return torch.cat([emb_sin, emb_cos], dim=1)  # (max_len, D/2)

    def get_2d_embed(self, h, w, ps=1):
        emb_v = self.pos_embed_1d[:h]
        emb_h = self.pos_embed_1d[:w]

        grid_v = emb_v.view(h, 1, self.hidden_size//2).repeat(1, w, 1)
        grid_h = emb_h.view(1, w, self.hidden_size//2).repeat(h, 1, 1)

        pos_embed = torch.cat([grid_h, grid_v], dim=-1) # h w c

        return rearrange(pos_embed, '(h p1) (w p2) c -> (h w p1 p2) c', p1=ps, p2=ps)

    def build_vision_encoder(self, config):
        encoder_config = config.encoder
        from modeling.vision_encoder.autoencoder import VQModel
        self.vision_encoder = VQModel(**encoder_config.params)
        self.vision_latent_dim = config.encoder.params.ddconfig.z_channels

    def build_llm_model(self, config):
        llm_config = config.llm
        llm_checkpoint = download(llm_config.checkpoint)
        if llm_config.type == "qwen3":
            from liger_kernel.transformers import apply_liger_kernel_to_qwen3
            print("apply liger kernel to qwen3")
            apply_liger_kernel_to_qwen3(rope=False)
            parallel_num = config.head.vision_pred.get("parallel_num", 1)
            from transformers import Qwen3Config
            tokenizer = AutoTokenizer.from_pretrained(llm_checkpoint)
            self.llm_config = Qwen3Config.from_pretrained(llm_checkpoint)
            if parallel_num > 1:
                print('parallel_num > 1, use Qwen3FlexWrapper')
                from modeling.llm.qwen3_navit import Qwen3FlexWrapper
                self.llm_model = Qwen3FlexWrapper.from_pretrained(
                llm_checkpoint,
                config=self.llm_config,
                use_packed_attn=True)
            else:
                print('parallel_num <= 1, use Qwen3ForCausalLMWrapper')
                from modeling.llm.qwen3_packed_wrapper import Qwen3ForCausalLMWrapper
                self.llm_model = Qwen3ForCausalLMWrapper.from_pretrained(
                llm_checkpoint,
                config=self.llm_config,
                use_packed_attn=True)
        else:
            raise ValueError(f"LLM type {llm_config.type} is not supported.")

        # add apecial tokens
        self.tokenizer = set_special_token_aliases(tokenizer)
        # add resolution special tokens: <|res_1|>, <|res_2|>, ..., <|res_256|>
        # <|res_1|>: 16, <|res_256|>: 4096
        self.tokenizer = add_resolution_special_tokens(
            self.tokenizer, max_resolution=4096, patch_size=config.vit_patch_size)
        
        # add query special tokens: <|query_1|>, <|query_2|>, ..., <|query_{parallel_num-1}|>
        self.tokenizer = add_query_special_tokens(
            self.tokenizer, parallel_num=config.head.vision_pred.get("parallel_num", 1))
        # resize token embeddings
        self.llm_model.resize_token_embeddings(len(self.tokenizer))

    def build_vision_head(self, config):
        head_config = config.head
        self.head_config = head_config
        self.hidden_size = self.llm_config.hidden_size
        self.vision_head_type = head_config.vision_pred.get("type", "standard")
        # build embed input
        if self.vision_head_type == "standard":
            self.vocab_size_vision = self.vision_encoder.codebook_size
            self.embed_tokens_vision = nn.Embedding(self.vocab_size_vision, self.hidden_size)
        else:
            self.embed_vision_mlp = MLPconnector(self.vision_latent_dim, self.hidden_size, "gelu_pytorch_tanh")
        
        # build prediction head
        model_dim = head_config.vision_pred.get("model_dim", 1024)
        if self.vision_head_type == "standard":
            self.vision_head = nn.Linear(self.hidden_size, self.vocab_size_vision)
        elif self.vision_head_type == "regression":
            self.vision_regression_head = nn.Sequential(
                nn.Linear(self.hidden_size, model_dim),
                nn.SiLU(),
                nn.Linear(model_dim, self.vision_latent_dim)
            )            
        elif self.vision_head_type == "gaussian_regression":
            self.vision_gaussian_head = nn.Sequential(
                nn.Linear(self.hidden_size, model_dim),
                nn.SiLU(),
                nn.Linear(model_dim, self.vision_latent_dim * 2)
            )
        elif self.vision_head_type == "diffusion_parallel_x":
            from modeling.vision_head.flow_head_parallel_x import DiffHead
            self.parallel_num = head_config.vision_pred.get("parallel_num", 1)
            self.ps = int(self.parallel_num ** 0.5)
            print(f'using difusion_parallel_x head with parallel_num={self.parallel_num}')
            self.vision_diffusion_head = DiffHead(
                ch_target=self.vision_latent_dim,
                ch_cond=self.hidden_size,
                ch_latent=model_dim,
                depth_latent=head_config.vision_pred.get("num_blocks", 3),
                depth_adanln=head_config.vision_pred.get("num_adaln", 1),
                time_shift=head_config.vision_pred.get("time_shift", 1.),
                time_schedule=head_config.vision_pred.get("time_schedule",'logit_normal'),
                P_mean=head_config.vision_pred.get("P_mean", 0.0),
                P_std=head_config.vision_pred.get("P_std", 1.0),
                parallel_num=self.parallel_num,
                diff_batch_mul=head_config.vision_pred.get("diff_batch_mul", 1),
                use_swiglu=head_config.vision_pred.get("use_swiglu", False),
            )
    
    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        # for ViT
        vit_image_tensors: List[torch.Tensor],
        vit_token_indexes_for_encoder: torch.LongTensor,
        packed_vit_rope_coords: torch.LongTensor,
        vit_token_seqlens: torch.LongTensor,
        vit_latent_shapes,
        gen_vit_latent_shapes,
        # for llm
        sequence_length: int,
        sample_lens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        # for loss
        ce_loss_indexes_text: torch.BoolTensor,
        packed_label_ids: torch.LongTensor,
        ce_loss_indexes_vision: torch.BoolTensor,
        packed_label_indexes_vision: torch.LongTensor,
        *args, **kwargs
    ) -> torch.Tensor:
        
        # Text embed
        packed_text_embedding = self.llm_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_empty(size=(sequence_length, self.hidden_size))
        packed_sequence.index_copy_(0, packed_text_indexes, packed_text_embedding)

        # Vision embed
        packed_vision_embedding, packed_vision_latents = self.encode_image(vit_image_tensors, packed_label_indexes_vision)
        packed_sequence.index_copy_(0, packed_vit_token_indexes, packed_vision_embedding.to(packed_sequence.dtype))
        packed_labels_vision = packed_vision_latents[packed_label_indexes_vision]

        split_lens = kwargs.get("split_lens", None)
        attn_modes = kwargs.get("attn_modes", None)
        if (split_lens is not None) and (attn_modes is not None):
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, self.parallel_num, packed_text_embedding.device)
            seqlen = sum(sample_lens.tolist())
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.llm_config.num_attention_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = None

        # LLM forward
        model_output = self.llm_model.model(
            inputs_embeds=packed_sequence.unsqueeze(0),
            position_ids=packed_position_ids.unsqueeze(0),
            sample_lens=sample_lens,
            attention_mask=attention_mask,
        )                
        llm_embed = model_output.last_hidden_state.squeeze(0)

        # compute 2d pos embed for diffusion head
        if len(ce_loss_indexes_vision) > 0 and "diffusion" in self.vision_head_type:
            with torch.no_grad():
                pos_embed_all = []
                for h, w in gen_vit_latent_shapes:
                    pos_embed_all.append(self.get_2d_embed(h, w, ps=self.ps if hasattr(self, 'ps') else 1))
                pos_embed_for_diff = torch.cat(pos_embed_all, dim=0)
            
        # Compute text and vision loss
        if len(ce_loss_indexes_text) > 0:
            logits_text = self.llm_model.lm_head(llm_embed[ce_loss_indexes_text])
            ce_loss_text = F.cross_entropy(logits_text, packed_label_ids, reduction="none")

        if len(ce_loss_indexes_vision) > 0:
            if self.vision_head_type == "standard":
                logits_vision = self.vision_head(llm_embed[ce_loss_indexes_vision])
                ce_loss_vision = F.cross_entropy(logits_vision, packed_labels_vision, reduction="none")
            elif self.vision_head_type == "regression":
                logits_vision = self.vision_regression_head(llm_embed[ce_loss_indexes_vision])
                ce_loss_vision = F.mse_loss(logits_vision, packed_labels_vision, reduction="none")
            elif self.vision_head_type == "gaussian_regression":
                raw_output = self.vision_gaussian_head(llm_embed[ce_loss_indexes_vision])
                logits_vision = gaussian_sample(raw_output)
                ce_loss_vision = F.mse_loss(logits_vision, packed_labels_vision, reduction="none")
            elif "diffusion_parallel" in self.vision_head_type:
                embed_loss = llm_embed[ce_loss_indexes_vision] + pos_embed_for_diff
                ce_loss_vision = self.vision_diffusion_head(
                    packed_labels_vision.view(-1, self.parallel_num, packed_labels_vision.shape[-1]).repeat(self.head_config.vision_pred.get('diff_batch_mul', 1), 1, 1),
                    embed_loss.view(-1, self.parallel_num, embed_loss.shape[-1]).repeat(self.head_config.vision_pred.get('diff_batch_mul', 1), 1, 1),
                )
            else:
                raise NotImplementedError

        return {
            'ce_loss_text': ce_loss_text,
            'ce_loss_vision': ce_loss_vision,
        }

    @torch.no_grad()
    def gen_image(self,
        cond_prompt,
        uncond_prompt=None,
        guidance_scale: float = 1.0,
        num_sampling_steps: int = 50,
        max_length: int = 64,
        num_images: int = 1,
        image_size = [256, 256],
        show_progress: bool = False,
    ):
        parallel_num = self.config.head.vision_pred.get("parallel_num", 1)
        if parallel_num > 1:
            return self.gen_image_block_causal(cond_prompt, uncond_prompt, guidance_scale, num_sampling_steps, max_length, num_images, image_size, show_progress)
        else:
            return self.gen_image_full_causal(cond_prompt, uncond_prompt, guidance_scale, num_sampling_steps, max_length, num_images, image_size, show_progress)

    @torch.no_grad()
    def gen_image_full_causal(self,
        cond_prompt,
        uncond_prompt=None,
        guidance_scale: float = 1.0,
        num_sampling_steps: int = 50,
        max_length: int = 64,
        num_images: int = 1,
        image_size = [256, 256],
        show_progress: bool = False,
    ):
        tokenizer = self.tokenizer
        device = self.device
        model = self.llm_model.model

        # determine whether to use parallel decoding
        is_parallel = ("diffusion_parallel" in self.vision_head_type)
        step_width = self.parallel_num if is_parallel else 1
        num_steps = max_length // step_width

        cond_ids = torch.tensor(tokenizer.encode(cond_prompt), device=device, dtype=torch.long)
        cond_emb = model.embed_tokens(cond_ids)
        if guidance_scale > 1.0:
            uncond_ids = torch.tensor(tokenizer.encode(uncond_prompt), device=device, dtype=torch.long)
            uncond_emb = model.embed_tokens(uncond_ids)

        img_start_id = tokenizer.start_of_image_id
        res_h_token_id = getattr(self.tokenizer, f"res_{image_size[0] // self.config.vit_patch_size}_id")
        res_w_token_id = getattr(self.tokenizer, f"res_{image_size[1] // self.config.vit_patch_size}_id")
        img_start_emb = model.embed_tokens(torch.tensor([img_start_id, res_h_token_id, res_w_token_id], device=device))

        h, w = image_size[0] // self.config.vit_patch_size, image_size[1] // self.config.vit_patch_size
        # prepare diff pos embed
        pos_embed_for_diff = self.get_2d_embed(h, w, ps=self.ps if hasattr(self, 'ps') else 1).unsqueeze(0)

        # add query tokens for parallel decoding
        if is_parallel:
            for i in range(1, self.parallel_num):
                query_token = torch.tensor([getattr(tokenizer, f"query_{i}_id")], device=self.device, dtype=torch.long)
                query_embed = self.llm_model.model.embed_tokens(query_token)
                img_start_emb = torch.cat([img_start_emb, query_embed], dim=0)

        input_embeds_cond = torch.cat(
            [cond_emb, img_start_emb], dim=0
        ).unsqueeze(0).repeat(num_images, 1, 1)
        outputs_c = model(
            inputs_embeds=input_embeds_cond,
            use_cache=True,
        )
        pkv_c = outputs_c.past_key_values

        if is_parallel:
            hidden_c = outputs_c.last_hidden_state[:, -step_width:] # [B, parallel_num, D]
        else:
            hidden_c = outputs_c.last_hidden_state[:, -1] # [B, D]

        if guidance_scale > 1.0:
            input_embeds_uncond = torch.cat(
                [uncond_emb, img_start_emb], dim=0
            ).unsqueeze(0).repeat(num_images, 1, 1)
            outputs_u = model(
                inputs_embeds=input_embeds_uncond,
                use_cache=True,
            )
            pkv_u = outputs_u.past_key_values
            if is_parallel:
                hidden_u = outputs_u.last_hidden_state[:, -step_width:] # [B, parallel_num, D]
            else:
                hidden_u = outputs_u.last_hidden_state[:, -1] # [B, D]

        out_tokens = []
        if show_progress:
            pbar = tqdm(total=num_steps, desc="Decoding Steps")
        for step in range(num_steps):
            if show_progress:
                pbar.update(1)
            h_fused = torch.cat([hidden_c, hidden_u], dim=0) if guidance_scale > 1.0 else hidden_c
            if not is_parallel:
                h_fused = h_fused + pos_embed_for_diff[:, step, :]
            else:
                h_fused = h_fused + pos_embed_for_diff[:, step*step_width:(step+1)*step_width, :]
            pred_latents = self.vision_diffusion_head.sample(h_fused, num_sampling_steps=num_sampling_steps, cfg=guidance_scale)
            # important! LFQ is used here
            curr_tokens = torch.sign(pred_latents)
            curr_embeds = self.embed_vision_mlp(curr_tokens)
            if not is_parallel:
                out_tokens.append(curr_tokens[:num_images].unsqueeze(1))
                model_input = curr_embeds.unsqueeze(1) # [B, D] -> [B, 1, D]
            else:
                out_tokens.append(curr_tokens[:num_images])
                model_input = curr_embeds # [B, N, D]

            # 2d pos embed
            model_input = model_input + pos_embed_for_diff[:, step*step_width:(step+1)*step_width, :]
            
            outputs_c = model(inputs_embeds=model_input[:num_images], past_key_values=pkv_c, use_cache=True)
            pkv_c = outputs_c.past_key_values
            if is_parallel:
                hidden_c = outputs_c.last_hidden_state[:, -step_width:] # [B, parallel_num, D]
            else:
                hidden_c = outputs_c.last_hidden_state[:, -1] # [B, D]
            if guidance_scale > 1.0:
                outputs_u = model(inputs_embeds=model_input[num_images:], past_key_values=pkv_u, use_cache=True)
                pkv_u = outputs_u.past_key_values
                if is_parallel:
                    hidden_u = outputs_u.last_hidden_state[:, -step_width:] # [B, parallel_num, D]
                else:
                    hidden_u = outputs_u.last_hidden_state[:, -1] # [B, D]
        full_output = torch.cat(out_tokens, dim=1)
        image = self.decode_image(full_output, [h, w], ps=self.ps if hasattr(self, 'ps') else 1) # [num_images, c, h, w]
        return image

    @torch.no_grad()
    def gen_image_block_causal(self,
        cond_prompt,
        uncond_prompt=None,
        guidance_scale: float = 1.0,
        num_sampling_steps: int = 50,
        max_length: int = 64,
        num_images: int = 1,
        image_size = [256, 256],
        show_progress: bool = False,
    ):
        tokenizer = self.tokenizer
        device = self.device
        model = self.llm_model.model

        step_width = self.parallel_num
        num_steps = max_length // step_width

        cond_ids = torch.tensor(tokenizer.encode(cond_prompt), device=device, dtype=torch.long)
        cond_emb = model.embed_tokens(cond_ids)
        if guidance_scale > 1.0:
            uncond_ids = torch.tensor(tokenizer.encode(uncond_prompt), device=device, dtype=torch.long)
            uncond_emb = model.embed_tokens(uncond_ids)

        img_start_id = tokenizer.start_of_image_id
        res_h_token_id = getattr(self.tokenizer, f"res_{image_size[0] // self.config.vit_patch_size}_id")
        res_w_token_id = getattr(self.tokenizer, f"res_{image_size[1] // self.config.vit_patch_size}_id")
        img_start_emb = model.embed_tokens(torch.tensor([img_start_id, res_h_token_id, res_w_token_id], device=device))

        h, w = image_size[0] // self.config.vit_patch_size, image_size[1] // self.config.vit_patch_size
        # prepare diff pos embed
        pos_embed_for_diff = self.get_2d_embed(h, w, ps=self.ps if hasattr(self, 'ps') else 1).unsqueeze(0)

        # add query tokens for parallel decoding
        for i in range(1, self.parallel_num):
            query_token = torch.tensor([getattr(tokenizer, f"query_{i}_id")], device=self.device, dtype=torch.long)
            query_embed = self.llm_model.model.embed_tokens(query_token)
            img_start_emb = torch.cat([img_start_emb, query_embed], dim=0)

        input_embeds_cond = torch.cat(
            [cond_emb, img_start_emb], dim=0
        ).unsqueeze(0).repeat(num_images, 1, 1)
        outputs_c = model(
            inputs_embeds=input_embeds_cond[:, :-step_width, :],
            use_cache=True,
        )
        pkv_c = outputs_c.past_key_values

        # bidirectional attn
        bi_attn_mask = torch.ones(
                (input_embeds_cond.shape[0], 1, step_width, step_width+pkv_c[0][0].shape[2]),
                dtype=torch.bool,
                device=device,
            )
        outputs_c = model(
            inputs_embeds=input_embeds_cond[:, -step_width:, :],
            past_key_values=pkv_c,
            use_cache=True,
            attention_mask=bi_attn_mask,
        )
        pkv_c = outputs_c.past_key_values
        hidden_c = outputs_c.last_hidden_state[:, -step_width:] # [B, parallel_num, D]

        if guidance_scale > 1.0:
            input_embeds_uncond = torch.cat(
                [uncond_emb, img_start_emb], dim=0
            ).unsqueeze(0).repeat(num_images, 1, 1)
            outputs_u = model(
                inputs_embeds=input_embeds_uncond[:, :-step_width, :],
                use_cache=True,
            )
            pkv_u = outputs_u.past_key_values
            outputs_u = model(
                inputs_embeds=input_embeds_uncond[:, -step_width:, :],
                past_key_values=pkv_u,
                use_cache=True,
                attention_mask=bi_attn_mask,
            )
            pkv_u = outputs_u.past_key_values
            hidden_u = outputs_u.last_hidden_state[:, -step_width:] # [B, parallel_num, D]

        out_tokens = []
        if show_progress:
            pbar = tqdm(total=num_steps, desc="Decoding Steps")
        for step in range(num_steps):
            if show_progress:
                pbar.update(1)
            h_fused = torch.cat([hidden_c, hidden_u], dim=0) if guidance_scale > 1.0 else hidden_c
            h_fused = h_fused + pos_embed_for_diff[:, step*step_width:(step+1)*step_width, :]
            pred_latents = self.vision_diffusion_head.sample(h_fused, num_sampling_steps=num_sampling_steps, cfg=guidance_scale)
            # important! LFQ is used here
            curr_tokens = torch.sign(pred_latents)
            curr_embeds = self.embed_vision_mlp(curr_tokens)
            out_tokens.append(curr_tokens[:num_images])
            model_input = curr_embeds # [B, N, D]
            # 2d pos embed
            model_input = model_input + pos_embed_for_diff[:, step*step_width:(step+1)*step_width, :]
 
            # bidirectional attn mask
            bi_attn_mask = torch.ones(
                (model_input.shape[0], 1, model_input.shape[1], model_input.shape[1]+pkv_c[0][0].shape[2]), 
                dtype=torch.bool,
                device=device
            )
            outputs_c = model(inputs_embeds=model_input[:num_images], past_key_values=pkv_c, use_cache=True, attention_mask=bi_attn_mask[:num_images])
            pkv_c = outputs_c.past_key_values
            hidden_c = outputs_c.last_hidden_state[:, -step_width:] # [B, parallel_num, D]

            if guidance_scale > 1.0:
                outputs_u = model(inputs_embeds=model_input[num_images:], past_key_values=pkv_u, use_cache=True, attention_mask=bi_attn_mask[num_images:])
                pkv_u = outputs_u.past_key_values
                hidden_u = outputs_u.last_hidden_state[:, -step_width:] # [B, parallel_num, D]

        full_output = torch.cat(out_tokens, dim=1)
        image = self.decode_image(full_output, [h, w], ps=self.ps if hasattr(self, 'ps') else 1) # [num_images, c, h, w]
        return image
    
    @torch.no_grad()
    def forward_inference(
        self, 
        sequence_plan,
        text_list, 
        image_list,
        ### HACK: modified to train mode
        do_sample: bool = True,
        max_length_text: int = 128,
        max_length_vision: int = 64,
        temperature: float = 1.0,
        sample_steps: int = 50,
        image_size = [256, 256],
        cfg_scale = 7.5,
        *args, **kwargs
    ):
        parallel_num = self.config.head.vision_pred.get("parallel_num", 1)
        if parallel_num > 1:
            return self.forward_inference_block_causal(sequence_plan, text_list, image_list, do_sample, max_length_text, max_length_vision, temperature, sample_steps, image_size, cfg_scale, *args, **kwargs)
        else:
            return self.forward_inference_full_causal(sequence_plan, text_list, image_list, do_sample, max_length_text, max_length_vision, temperature, sample_steps, image_size, cfg_scale, *args, **kwargs)

    @torch.no_grad()
    def forward_inference_full_causal(
        self, 
        sequence_plan,
        text_list, 
        image_list,
        ### HACK: modified to train mode
        do_sample: bool = True,
        max_length_text: int = 128,
        max_length_vision: int = 64,
        temperature: float = 1.0,
        sample_steps: int = 50,
        image_size = [256, 256],
        cfg_scale = 7.5,
        *args, **kwargs
    ):
        tokenizer = self.tokenizer
        past_key_values = None
        past_key_values_un = None
        generated_sequence = {
            "generated_text": [],
            "generated_image": [],
        }
        use_cfg = cfg_scale > 1.0
        context_embed = []
        context_embed_un = []
        for item in sequence_plan:
            # input_embedding & output_head
            cur_item_type = item['type']
            if cur_item_type == 'text':
                start_token = torch.tensor([tokenizer.im_start_id], device=self.device, dtype=torch.long)
                end_token_id = tokenizer.im_end_id
                max_length = max_length_text
            elif cur_item_type == 'image':
                res_h_token_id = getattr(self.tokenizer, f"res_{image_size[0] // self.config.vit_patch_size}_id")
                res_w_token_id = getattr(self.tokenizer, f"res_{image_size[1] // self.config.vit_patch_size}_id")

                start_token = torch.tensor([tokenizer.start_of_image_id, res_h_token_id, res_w_token_id], device=self.device, dtype=torch.long)
                end_token_id = tokenizer.end_of_image_id
                max_length = max_length_vision
            
                start_embed = self.llm_model.model.embed_tokens(start_token)
                context_embed.append(start_embed)
                context_embed_un.append(start_embed)

            if item["from"] == "model":
                # generate
                step = 0
                # prepare diff pos embed
                if cur_item_type == "image" and "diffusion" in self.vision_head_type:
                    h, w = image_size[0] // self.config.vit_patch_size, image_size[1] // self.config.vit_patch_size
                    pos_embed_for_diff = self.get_2d_embed(h, w, ps=self.ps if hasattr(self, 'ps') else 1).unsqueeze(0)
                    
                # determine whether to use parallel decoding
                is_parallel = (cur_item_type == 'image' and "diffusion_parallel" in self.vision_head_type)
                step_width = self.parallel_num if is_parallel else 1
                # add query tokens for parallel decoding
                if is_parallel:
                    for i in range(1, self.parallel_num):
                        query_token = torch.tensor([getattr(tokenizer, f"query_{i}_id")], device=self.device, dtype=torch.long)
                        query_embed = self.llm_model.model.embed_tokens(query_token)
                        context_embed.append(query_embed)
                        context_embed_un.append(query_embed)

                curr_embeds = torch.cat(context_embed, dim=0)   # （l, d)
                if use_cfg: curr_embeds_un = torch.cat(context_embed_un, dim=0)   # （l, d)
                out_tokens = []
                while step < max_length:
                    model_input = curr_embeds[None] if curr_embeds.dim() == 2 else curr_embeds
                    if cur_item_type == "image" and step > 0:
                        model_input = model_input + pos_embed_for_diff[:, step-step_width:step, :]
                    outputs = self.llm_model.model(
                        inputs_embeds=model_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    if use_cfg:
                        model_input_un = curr_embeds_un[None] if curr_embeds_un.dim() == 2 else curr_embeds_un
                        if cur_item_type == "image" and step > 0:
                            model_input_un = model_input_un + pos_embed_for_diff[:, step-step_width:step, :]
                        outputs_un = self.llm_model.model(
                            inputs_embeds=model_input_un,
                            past_key_values=past_key_values_un,
                            use_cache=True,
                        )
                        past_key_values_un = outputs_un.past_key_values

                    if not is_parallel:
                        hidden_state = outputs.last_hidden_state[:, -1]   # (1, d): only decode one token
                        if use_cfg: hidden_state_un = outputs_un.last_hidden_state[:, -1]   # (1, d): only decode one token
                    else:
                        hidden_state = outputs.last_hidden_state[:, -step_width:]   # (1, parallel_num, d): decode parallel_num tokens
                        if use_cfg: hidden_state_un = outputs_un.last_hidden_state[:, -step_width:]   # (1, parallel_num, d): decode parallel_num tokens
                    
                    hidden_state_fused = torch.cat([hidden_state, hidden_state_un], dim=0) if use_cfg else hidden_state

                    if cur_item_type == "image" and self.vision_head_type != "standard":
                        if self.vision_head_type == "regression":                        
                            pred_latents = self.vision_regression_head(hidden_state)
                        elif self.vision_head_type == "gaussian_regression":   
                            raw_output = self.vision_gaussian_head(hidden_state)
                            pred_latents = gaussian_sample(raw_output)
                        elif "diffusion" in self.vision_head_type:
                            if not is_parallel:
                                hidden_state_fused = hidden_state_fused + pos_embed_for_diff[:, step, :]
                            else:
                                hidden_state_fused = hidden_state_fused + pos_embed_for_diff[:, step:step+step_width, :]
                            pred_latents = self.vision_diffusion_head.sample(hidden_state_fused, num_sampling_steps=sample_steps, cfg=cfg_scale)
                        # important! LFQ is used here
                        curr_tokens_fused = torch.sign(pred_latents)
                        curr_tokens = curr_tokens_fused[:1]
                        curr_embeds_fused = self.embed_vision_mlp(curr_tokens_fused)
                        curr_embeds = curr_embeds_fused[:1]
                        if use_cfg: curr_embeds_un = curr_embeds_fused[1:]
                    else:
                        head = self.llm_model.lm_head if cur_item_type == 'text' else self.vision_head
                        pred_logits = head(hidden_state)
                        codebook = (
                            self.llm_model.model.embed_tokens
                            if cur_item_type == "text"
                            else self.embed_tokens_vision
                        )
                        curr_tokens, curr_embeds = sample_codebook(
                            pred_logits, cur_item_type, codebook,
                            do_sample = True,
                            temperature = 1.0,
                            top_k = 1200,
                            top_p = 0.95
                        )
                        
                    step += step_width
                    out_tokens.append(curr_tokens)
                    if curr_tokens.nelement() == 1 and curr_tokens.item() == end_token_id:
                        context_embed = []
                        break

                full_output = torch.cat(out_tokens, dim=0) if out_tokens[0].dim() == 2 else torch.cat(out_tokens, dim=1)
                if cur_item_type == 'text':     
                    tokens = tokenizer.convert_ids_to_tokens(full_output, skip_special_tokens=True)
                    text = tokenizer.convert_tokens_to_string([tok for tok in tokens if tok is not None])
                    generated_sequence["generated_text"].append(text)
                else:
                    image = self.decode_image(full_output[None] if full_output.dim() == 2 else full_output, ps=self.ps if hasattr(self, 'ps') else 1)
                    generated_sequence["generated_image"].append(image)
            else:
                # update & prefill context 
                if cur_item_type == 'text':
                    text = text_list.pop(0)
                    text_ids = torch.tensor(tokenizer.encode(text)).to(self.device)
                    prefill_embedding = self.llm_model.model.embed_tokens(text_ids)
                    context_embed.append(prefill_embedding)
                    if use_cfg:
                        text_un = remove_first_user_block(text)
                        text_ids_un = torch.tensor(tokenizer.encode(text_un)).to(self.device)
                        prefill_embedding_un = self.llm_model.model.embed_tokens(text_ids_un)
                        context_embed_un.append(prefill_embedding_un)
                elif cur_item_type == 'image':
                    vit_image_tensor = image_list.pop(0).to(self.device)
                    prefill_embedding = self.encode_image([vit_image_tensor])[0]
                    end_token = torch.tensor([end_token_id], device=self.device, dtype=torch.long)
                    end_embed = self.llm_model.model.embed_tokens(end_token)
                    context_embed.append(prefill_embedding)
                    context_embed.append(end_embed)
                    if use_cfg:
                        context_embed_un.append(prefill_embedding)
                        context_embed_un.append(end_embed)

        return generated_sequence

    @torch.no_grad()
    def forward_inference_block_causal(
        self, 
        sequence_plan,
        text_list, 
        image_list,
        ### HACK: modified to train mode
        do_sample: bool = True,
        max_length_text: int = 128,
        max_length_vision: int = 64,
        temperature: float = 1.0,
        sample_steps: int = 50,
        image_size = [256, 256],
        cfg_scale = 7.5,
        *args, **kwargs
    ):
        tokenizer = self.tokenizer
        past_key_values = None
        past_key_values_un = None
        generated_sequence = {
            "generated_text": [],
            "generated_image": [],
        }
        use_cfg = cfg_scale > 1.0
        context_embed = []
        context_embed_un = []
        for item in sequence_plan:
            # input_embedding & output_head
            cur_item_type = item['type']
            if cur_item_type == 'text':
                start_token = torch.tensor([tokenizer.im_start_id], device=self.device, dtype=torch.long)
                end_token_id = tokenizer.im_end_id
                max_length = max_length_text
            elif cur_item_type == 'image':
                res_h_token_id = getattr(self.tokenizer, f"res_{image_size[0] // self.config.vit_patch_size}_id")
                res_w_token_id = getattr(self.tokenizer, f"res_{image_size[1] // self.config.vit_patch_size}_id")

                start_token = torch.tensor([tokenizer.start_of_image_id, res_h_token_id, res_w_token_id], device=self.device, dtype=torch.long)
                end_token_id = tokenizer.end_of_image_id
                max_length = max_length_vision
            
                start_embed = self.llm_model.model.embed_tokens(start_token)
                context_embed.append(start_embed)
                context_embed_un.append(start_embed)

            if item["from"] == "model":
                # generate
                step = 0
                # prepare diff pos embed
                if cur_item_type == "image" and "diffusion" in self.vision_head_type:
                    h, w = image_size[0] // self.config.vit_patch_size, image_size[1] // self.config.vit_patch_size
                    pos_embed_for_diff = self.get_2d_embed(h, w, ps=self.ps if hasattr(self, 'ps') else 1).unsqueeze(0)

                is_parallel = (cur_item_type == 'image' and "diffusion_parallel" in self.vision_head_type)
                step_width = self.parallel_num if is_parallel else 1
    
                # add query tokens for parallel decoding
                if is_parallel:
                    for i in range(1, self.parallel_num):
                        query_token = torch.tensor([getattr(tokenizer, f"query_{i}_id")], device=self.device, dtype=torch.long)
                        query_embed = self.llm_model.model.embed_tokens(query_token)
                        context_embed.append(query_embed)
                        context_embed_un.append(query_embed)

                curr_embeds = torch.cat(context_embed, dim=0)[None]   # （1, l, d)
                if use_cfg: curr_embeds_un = torch.cat(context_embed_un, dim=0)[None]   # （1, l, d)
                out_tokens = []
                while step < max_length:
                    if step == 0 and is_parallel:
                        outputs = self.llm_model.model(
                            inputs_embeds=curr_embeds[:, :-step_width, :],
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                        past_key_values = outputs.past_key_values
                        bi_attn_mask = torch.ones(
                            (curr_embeds.shape[0], 1, step_width, step_width+past_key_values[0][0].shape[2]),
                            dtype=torch.bool,
                            device=self.device,
                        )
                        outputs = self.llm_model.model(
                            inputs_embeds=curr_embeds[:, -step_width:, :],
                            past_key_values=past_key_values,
                            use_cache=True,
                            attention_mask=bi_attn_mask,
                        )
                        past_key_values = outputs.past_key_values
                        if use_cfg:
                            outputs_un = self.llm_model.model(
                                inputs_embeds=curr_embeds_un[:, :-step_width, :],
                                past_key_values=past_key_values_un,
                                use_cache=True,
                            )
                            past_key_values_un = outputs_un.past_key_values
                            outputs_un = self.llm_model.model(
                                inputs_embeds=curr_embeds_un[:, -step_width:, :],
                                past_key_values=past_key_values_un,
                                use_cache=True,
                                attention_mask=bi_attn_mask,
                            )
                            past_key_values_un = outputs_un.past_key_values
                    else:
                        bi_attn_mask = torch.ones(
                            (curr_embeds.shape[0], 1, step_width, step_width+past_key_values[0][0].shape[2]),
                            dtype=torch.bool,
                            device=self.device,
                        )
                        model_input = curr_embeds + pos_embed_for_diff[:, step-step_width:step, :] if is_parallel else curr_embeds
                        outputs = self.llm_model.model(
                            inputs_embeds=model_input,
                            past_key_values=past_key_values,
                            use_cache=True,
                            attention_mask=bi_attn_mask if is_parallel else None,
                        )
                        past_key_values = outputs.past_key_values
                        if use_cfg:
                            model_input_un = curr_embeds_un + pos_embed_for_diff[:, step-step_width:step, :] if is_parallel else curr_embeds_un
                            outputs_un = self.llm_model.model(
                                inputs_embeds=model_input_un,
                                past_key_values=past_key_values_un,
                                use_cache=True,
                                attention_mask=bi_attn_mask if is_parallel else None,
                            )
                            past_key_values_un = outputs_un.past_key_values

                    if not is_parallel:
                        hidden_state = outputs.last_hidden_state[:, -1]   # (1, d): only decode one token
                        if use_cfg: hidden_state_un = outputs_un.last_hidden_state[:, -1]   # (1, d): only decode one token
                    else:
                        hidden_state = outputs.last_hidden_state[:, -step_width:]   # (1, parallel_num, d): decode parallel_num tokens
                        if use_cfg: hidden_state_un = outputs_un.last_hidden_state[:, -step_width:]   # (1, parallel_num, d): decode parallel_num tokens
                    
                    hidden_state_fused = torch.cat([hidden_state, hidden_state_un], dim=0) if use_cfg else hidden_state

                    if cur_item_type == "image" and self.vision_head_type != "standard":
                        if self.vision_head_type == "regression":                        
                            pred_latents = self.vision_regression_head(hidden_state)
                        elif self.vision_head_type == "gaussian_regression":   
                            raw_output = self.vision_gaussian_head(hidden_state)
                            pred_latents = gaussian_sample(raw_output)
                        elif "diffusion" in self.vision_head_type:
                            hidden_state_fused = hidden_state_fused + pos_embed_for_diff[:, step:step+step_width, :]
                            pred_latents = self.vision_diffusion_head.sample(hidden_state_fused, num_sampling_steps=sample_steps, cfg=cfg_scale)
                        # important! LFQ is used here
                        curr_tokens_fused = torch.sign(pred_latents)
                        curr_tokens = curr_tokens_fused[:1]
                        curr_embeds_fused = self.embed_vision_mlp(curr_tokens_fused)
                        curr_embeds = curr_embeds_fused[:1]
                        if use_cfg: curr_embeds_un = curr_embeds_fused[1:]
                    else:
                        head = self.llm_model.lm_head if cur_item_type == 'text' else self.vision_head
                        pred_logits = head(hidden_state)
                        codebook = (
                            self.llm_model.model.embed_tokens
                            if cur_item_type == "text"
                            else self.embed_tokens_vision
                        )
                        curr_tokens, curr_embeds = sample_codebook(
                            pred_logits, cur_item_type, codebook,
                            do_sample = True,
                            temperature = 1.0,
                            top_k = 1200,
                            top_p = 0.95
                        )
                        
                    step += step_width
                    out_tokens.append(curr_tokens)
                    if curr_tokens.nelement() == 1 and curr_tokens.item() == end_token_id:
                        context_embed = []
                        break

                full_output = torch.cat(out_tokens, dim=0) if out_tokens[0].dim() == 2 else torch.cat(out_tokens, dim=1)
                if cur_item_type == 'text':     
                    tokens = tokenizer.convert_ids_to_tokens(full_output, skip_special_tokens=True)
                    text = tokenizer.convert_tokens_to_string([tok for tok in tokens if tok is not None])
                    generated_sequence["generated_text"].append(text)
                else:
                    image = self.decode_image(full_output[None] if full_output.dim() == 2 else full_output, ps=self.ps if hasattr(self, 'ps') else 1)
                    generated_sequence["generated_image"].append(image)
            else:
                # update & prefill context 
                if cur_item_type == 'text':
                    text = text_list.pop(0)
                    text_ids = torch.tensor(tokenizer.encode(text)).to(self.device)
                    prefill_embedding = self.llm_model.model.embed_tokens(text_ids)
                    context_embed.append(prefill_embedding)
                    if use_cfg:
                        text_un = remove_first_user_block(text)
                        text_ids_un = torch.tensor(tokenizer.encode(text_un)).to(self.device)
                        prefill_embedding_un = self.llm_model.model.embed_tokens(text_ids_un)
                        context_embed_un.append(prefill_embedding_un)
                elif cur_item_type == 'image':
                    vit_image_tensor = image_list.pop(0).to(self.device)
                    prefill_embedding = self.encode_image([vit_image_tensor])[0]
                    end_token = torch.tensor([end_token_id], device=self.device, dtype=torch.long)
                    end_embed = self.llm_model.model.embed_tokens(end_token)
                    context_embed.append(prefill_embedding)
                    context_embed.append(end_embed)
                    if use_cfg:
                        context_embed_un.append(prefill_embedding)
                        context_embed_un.append(end_embed)

        return generated_sequence

    def encode_image(self, image_list, packed_label_indexes_vision=None):
        with torch.no_grad():
            # switch vt_forward_func to speed up encoding
            if self.config.encoder.get("vt_forward_func", "group") == "maxpad":
                vt_forward_func = self.vision_encoder.vt_forward_maxpad
            else:
                vt_forward_func = self.vision_encoder.vt_forward
            packed_vision_latents = vt_forward_func(
                image_list=image_list, max_bs=self.config.encoder.get("max_bs", 32), ps=self.ps if hasattr(self, 'ps') else 1)

        if self.training and packed_label_indexes_vision is not None:
            # perturb the vision latents during training for better generalization
            # only perturb vision tokens to be generated
            packed_vision_latents_flip = packed_vision_latents.clone().detach()
            packed_vision_latents_flip[packed_label_indexes_vision] = flip_tensor_elements_uniform_prob(
                    packed_vision_latents_flip[packed_label_indexes_vision],
                    self.head_config.get("vision_perturb", 0.0))
            packed_vision_embedding = self.embed_vision_mlp(packed_vision_latents_flip)
        else:
            packed_vision_embedding = self.embed_vision_mlp(packed_vision_latents)
            
        # add 2d position embedding
        img_shapes = []
        for img in image_list: img_shapes.append(img.shape[-2:])
        with torch.no_grad():
            pos_embed_for_vae = []
            for h, w in img_shapes:
                pos_embed_for_vae.append(self.get_2d_embed(h // self.config.vit_patch_size, w // self.config.vit_patch_size, ps=self.ps if hasattr(self, 'ps') else 1))
            pos_embed_for_vae = torch.cat(pos_embed_for_vae, dim=0)
            packed_vision_embedding += pos_embed_for_vae

        return packed_vision_embedding, packed_vision_latents.clone().detach()

    def decode_image(self, image_latents, image_size=None, ps=1):
        if image_size is None:
            h = w = int(image_latents.size(1) ** 0.5)
        else:
            h, w = image_size
            
        image_latents = rearrange(image_latents, 'b (h w p1 p2) c -> b c (h p1) (w p2)', h=h//ps, w=w//ps, p1=ps, p2=ps)
        output = self.vision_encoder.decode(image_latents)     # [1, c, h, w]
        
        return output