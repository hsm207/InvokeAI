from dataclasses import dataclass
from typing import List, Optional, cast

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from invokeai.backend.ip_adapter.ip_attention_weights import IPAttentionProcessorWeights
from invokeai.backend.stable_diffusion.diffusion.regional_ip_data import RegionalIPData
from invokeai.backend.stable_diffusion.diffusion.regional_prompt_data import RegionalPromptData
from invokeai.backend.util.logging import InvokeAILogger # Import the logger

import os

# Get a logger instance for this module
logger = InvokeAILogger.get_logger(__name__)

def detailed_tensor_diagnostics(name, tensor):
    """Log detailed diagnostic information about a tensor at DEBUG level."""
    logger.debug(f"\n=== {name} ===")
    logger.debug(f"Shape: {tensor.shape}")
    logger.debug(f"Dtype: {tensor.dtype}")
    logger.debug(f"Device: {tensor.device}")
    logger.debug(f"Contiguous: {tensor.is_contiguous()}")
    logger.debug(f"Requires grad: {tensor.requires_grad}")
    logger.debug(f"Storage offset: {tensor.storage_offset()}")
    logger.debug(f"Stride: {tensor.stride()}")
    
    # Check for NaNs or extreme values
    if tensor.dtype.is_floating_point:
        logger.debug(f"Contains NaN: {torch.isnan(tensor).any().item()}")
        logger.debug(f"Min value: {tensor.min().item()}")
        logger.debug(f"Max value: {tensor.max().item()}")
    
    # Add a hash of the first few values to check content
    sample = tensor.flatten()[:10].cpu().tolist()
    logger.debug(f"First 10 values sample: {sample}")

def chunked_scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, 
                                         is_causal=False, chunk_size_override: Optional[int] = None):
    """Process attention in chunks to avoid Pytorch MPS backend errors with large sequence lengths.

    See https://github.com/pytorch/pytorch/pull/149268 for details.

    If running on MPS, automatically estimates an optimal chunk size based on memory constraints.
    The `chunk_size_override` parameter can be used to force a specific chunk size, but this is
    generally not recommended on MPS unless you know the optimal value for your hardware.
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    key_len = key.shape[2]
    
    # Determine chunk size
    if query.device.type == 'mps':
        if chunk_size_override is not None:
            chunk_size = chunk_size_override
        else:
            # Estimate chunk size based on available memory (heuristic)
            chunk_size = estimate_optimal_chunk_size(
                batch_size, num_heads, seq_len, key_len, head_dim, query.dtype
            )
    else:
        # For non-MPS devices, use a large default or the override if provided
        chunk_size = chunk_size_override if chunk_size_override is not None else seq_len

    # Skip chunking if sequence length is already small enough
    if seq_len <= chunk_size:
        logger.info(f"[Chunking Skipped] seq_len {seq_len} <= chunk_size {chunk_size}") # Removed commented print
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
        )

    # Process in chunks
    logger.info(f"[Chunking Active] Processing seq_len {seq_len} in chunks of {chunk_size}") # Replaced print with logger.debug
    chunks = []
    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        query_chunk = query[:, :, chunk_start:chunk_end, :]
        # Properly chunk the attention mask if needed
        chunk_attn_mask = attn_mask
        if attn_mask is not None and attn_mask.shape[2] != 1:
            chunk_attn_mask = attn_mask[:, :, chunk_start:chunk_end, :]
        chunk_output = F.scaled_dot_product_attention(
            query_chunk, key, value, attn_mask=chunk_attn_mask, 
            dropout_p=dropout_p, is_causal=is_causal
        )
        chunks.append(chunk_output)
    return torch.cat(chunks, dim=2)

def estimate_optimal_chunk_size(
    batch_size, num_heads, seq_len, key_len, head_dim, dtype, max_mem_bytes=8*1024*1024
):
    """
    Estimate the largest chunk size for attention that fits within max_mem_bytes.
    Considers the largest intermediate tensors:
      - attention scores [batch, heads, chunk_size, key_len]
      - softmax output [batch, heads, chunk_size, key_len]
      - output [batch, heads, chunk_size, head_dim]

    Args:
        batch_size: Batch size of the input tensors.
        num_heads: Number of attention heads.
        seq_len: Sequence length of the query tensor.
        key_len: Sequence length of the key/value tensors.
        head_dim: Dimension of each attention head.
        dtype: Data type of the tensors (e.g., torch.float16).
        max_mem_bytes: Maximum estimated memory (in bytes) allowed for the sum of
                       the largest intermediate tensors per chunk. Defaults to 8MB.
    Returns:
        An estimated optimal chunk size (integer).
    """
    dtype_size = torch.tensor([], dtype=dtype).element_size()
    if batch_size == 0 or num_heads == 0 or key_len == 0 or head_dim == 0:
        return 1

    # Let x = chunk_size
    # Total memory per chunk (in bytes):
    #   2 * [batch, heads, x, key_len] (scores + softmax)
    # + 1 * [batch, heads, x, head_dim] (output)
    # = 2 * batch_size * num_heads * x * key_len * dtype_size
    #   + batch_size * num_heads * x * head_dim * dtype_size
    # = batch_size * num_heads * x * (2 * key_len + head_dim) * dtype_size

    denom = batch_size * num_heads * (2 * key_len + head_dim) * dtype_size
    if denom == 0:
        return 1

    max_chunk = max_mem_bytes // denom

    chunk_size = max(1, min(seq_len, int(max_chunk)))
    aligned_chunk_size = (chunk_size // 8) * 8
    return max(1, aligned_chunk_size)

@dataclass
class IPAdapterAttentionWeights:
    ip_adapter_weights: IPAttentionProcessorWeights
    skip: bool
    negative: bool


class CustomAttnProcessor2_0(AttnProcessor2_0):
    """A custom implementation of AttnProcessor2_0 that supports additional Invoke features.
    This implementation is based on
    https://github.com/huggingface/diffusers/blame/fcfa270fbd1dc294e2f3a505bae6bcb791d721c3/src/diffusers/models/attention_processor.py#L1204
    Supported custom features:
    - IP-Adapter
    - Regional prompt attention
    """

    def __init__(
        self,
        ip_adapter_attention_weights: Optional[List[IPAdapterAttentionWeights]] = None,
    ):
        """Initialize a CustomAttnProcessor2_0.
        Note: Arguments that are the same for all attention layers are passed to __call__(). Arguments that are
        layer-specific are passed to __init__().
        Args:
            ip_adapter_weights: The IP-Adapter attention weights. ip_adapter_weights[i] contains the attention weights
                for the i'th IP-Adapter.
        """
        super().__init__()
        self._ip_adapter_attention_weights = ip_adapter_attention_weights

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        # For Regional Prompting:
        regional_prompt_data: Optional[RegionalPromptData] = None,
        percent_through: Optional[torch.Tensor] = None,
        # For IP-Adapter:
        regional_ip_data: Optional[RegionalIPData] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        """Apply attention.
        Args:
            regional_prompt_data: The regional prompt data for the current batch. If not None, this will be used to
                apply regional prompt masking.
            regional_ip_data: The IP-Adapter data for the current batch.
        """
        # If true, we are doing cross-attention, if false we are doing self-attention.
        is_cross_attention = encoder_hidden_states is not None

        # Start unmodified block from AttnProcessor2_0.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # End unmodified block from AttnProcessor2_0.

        _, query_seq_len, _ = hidden_states.shape
        # Handle regional prompt attention masks.
        if regional_prompt_data is not None and is_cross_attention:
            assert percent_through is not None
            prompt_region_attention_mask = regional_prompt_data.get_cross_attn_mask(
                query_seq_len=query_seq_len, key_seq_len=sequence_length
            )

            if attention_mask is None:
                attention_mask = prompt_region_attention_mask
            else:
                attention_mask = prompt_region_attention_mask + attention_mask

        # Start unmodified block from AttnProcessor2_0.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # ---------------------
        logger.debug("\n----- DIAGNOSTICS FOR APPLICATION ENVIRONMENT -----")
        detailed_tensor_diagnostics("query", query)
        detailed_tensor_diagnostics("key", key)
        detailed_tensor_diagnostics("value", value)
        if attention_mask is not None:
            detailed_tensor_diagnostics("attention_mask", attention_mask)
    
        # Save tensors for testing
        tensors = {
            "query": query.detach().cpu(),
            "key": key.detach().cpu(),
            "value": value.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu() if attention_mask is not None else None
        }
    
        # os.makedirs(os.path.join(os.getcwd(), 'tests'), exist_ok=True)
        # torch.save(tensors, os.path.join(os.getcwd(), 'tests', 'problematic_tensors.pt'))
        # logger.debug(f"Saved tensors to {os.path.join(os.getcwd(), 'tests', 'problematic_tensors.pt')}")
        
        # ---------------------
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = chunked_scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, 
            dropout_p=0.0, is_causal=False
        )
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # End unmodified block from AttnProcessor2_0.

        # Apply IP-Adapter conditioning.
        if is_cross_attention:
            if self._ip_adapter_attention_weights:
                assert regional_ip_data is not None
                ip_masks = regional_ip_data.get_masks(query_seq_len=query_seq_len)

                assert (
                    len(regional_ip_data.image_prompt_embeds)
                    == len(self._ip_adapter_attention_weights)
                    == len(regional_ip_data.scales)
                    == ip_masks.shape[1]
                )

                for ipa_index, ipa_embed in enumerate(regional_ip_data.image_prompt_embeds):
                    ipa_weights = self._ip_adapter_attention_weights[ipa_index].ip_adapter_weights
                    ipa_scale = regional_ip_data.scales[ipa_index]
                    ip_mask = ip_masks[0, ipa_index, ...]

                    # The batch dimensions should match.
                    assert ipa_embed.shape[0] == encoder_hidden_states.shape[0]
                    # The token_len dimensions should match.
                    assert ipa_embed.shape[-1] == encoder_hidden_states.shape[-1]

                    ip_hidden_states = ipa_embed

                    # Expected ip_hidden_state shape: (batch_size, num_ip_images, ip_seq_len, ip_image_embedding)

                    if not self._ip_adapter_attention_weights[ipa_index].skip:
                        # apply the IP-Adapter weights to the negative embeds
                        if self._ip_adapter_attention_weights[ipa_index].negative:
                            ip_hidden_states = torch.cat([ip_hidden_states[1], ip_hidden_states[0] * 0], dim=0)

                        ip_key = ipa_weights.to_k_ip(ip_hidden_states)
                        ip_value = ipa_weights.to_v_ip(ip_hidden_states)

                        # Expected ip_key and ip_value shape:
                        # (batch_size, num_ip_images, ip_seq_len, head_dim * num_heads)

                        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                        # Expected ip_key and ip_value shape:
                        # (batch_size, num_heads, num_ip_images * ip_seq_len, head_dim)

                        # TODO: add support for attn.scale when we move to Torch 2.1
                        ip_hidden_states = F.scaled_dot_product_attention(
                            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                        )

                        # Expected ip_hidden_states shape: (batch_size, num_heads, query_seq_len, head_dim)
                        ip_hidden_states = ip_hidden_states.transpose(1, 2).reshape(
                            batch_size, -1, attn.heads * head_dim
                        )

                        ip_hidden_states = ip_hidden_states.to(query.dtype)

                        # Expected ip_hidden_states shape: (batch_size, query_seq_len, num_heads * head_dim)
                        hidden_states = hidden_states + ipa_scale * ip_hidden_states * ip_mask
            else:
                # If IP-Adapter is not enabled, then regional_ip_data should not be passed in.
                assert regional_ip_data is None

        # Start unmodified block from AttnProcessor2_0.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # End of unmodified block from AttnProcessor2_0

        # casting torch.Tensor to torch.FloatTensor to avoid type issues
        return cast(torch.FloatTensor, hidden_states)
