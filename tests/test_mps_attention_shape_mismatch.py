import torch
import torch.nn.functional as F
import pytest
import os
import logging
from invokeai.backend.stable_diffusion.diffusion.custom_atttention import chunked_scaled_dot_product_attention, detailed_tensor_diagnostics

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detailed_tensor_diagnostics(name, tensor):
    """Print detailed diagnostic information about a tensor."""
    print(f"\n=== {name} ===")
    print(f"Shape: {tensor.shape}")
    print(f"Dtype: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Contiguous: {tensor.is_contiguous()}")
    print(f"Requires grad: {tensor.requires_grad}")
    print(f"Storage offset: {tensor.storage_offset()}")
    print(f"Stride: {tensor.stride()}")
    
    # Check for NaNs or extreme values
    if tensor.dtype.is_floating_point:
        print(f"Contains NaN: {torch.isnan(tensor).any().item()}")
        print(f"Min value: {tensor.min().item()}")
        print(f"Max value: {tensor.max().item()}")
    
    # Add a hash of the first few values to check content
    sample = tensor.flatten()[:10].cpu().tolist()
    print(f"First 10 values sample: {sample}")

def test_mps_attention_shape_mismatch():
    """Test that demonstrates the MPS error with specific large sequence dimensions."""
    
    # Skip test if MPS is not available
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available, skipping test")
    
     # Load saved tensors from disk
    tensor_path = os.path.join(os.getcwd(), 'tests', 'problematic_tensors.pt')
    
    if not os.path.exists(tensor_path):
        pytest.skip(f"Test data file {tensor_path} not found")
    
    print(f"Loading tensors from {tensor_path}")
    saved_data = torch.load(tensor_path)
    
    query = saved_data["query"]
    key = saved_data["key"]
    value = saved_data["value"]
    attention_mask = saved_data["attention_mask"] if "attention_mask" in saved_data else None
    
    # First verify on CPU (this should work fine)
    print("Testing on CPU...")
    cpu_output = F.scaled_dot_product_attention(
        query.to("cpu"), 
        key.to("cpu"), 
        value.to("cpu"), 
        attn_mask=attention_mask.to("cpu"),
        dropout_p=0.0, 
        is_causal=False
    )
    print(f"CPU output shape: {cpu_output.shape}")

    print("\n----- DIAGNOSTICS FOR TEST ENVIRONMENT -----")
    query_mps = query.to("mps")
    key_mps = key.to("mps")
    value_mps = value.to("mps")
    mask_mps = attention_mask.to("mps") if attention_mask is not None else None

    detailed_tensor_diagnostics("query", query_mps)
    detailed_tensor_diagnostics("key", key_mps)
    detailed_tensor_diagnostics("value", value_mps)
    if mask_mps is not None:
        detailed_tensor_diagnostics("attention_mask", mask_mps)
    
    # Test on MPS (should fail with dimension mismatch)
    print("Testing on MPS...")
    try:
        mps_output = F.scaled_dot_product_attention(
            query_mps,
            key_mps,
            value_mps,
            attn_mask=mask_mps,
            dropout_p=0.0,
            is_causal=False
        )
        print("MPS execution succeeded unexpectedly!")
        assert False, "Expected MPS to fail but it didn't"
    except Exception as e:
        print(f"MPS failed as expected with error: {str(e)}")
        print("Test passed: MPS correctly fails with large sequence dimensions")
        # Verify the error message contains expected text about placeholder shape mismatch
        assert "Placeholder shape mismatches" in str(e) or "failed assertion" in str(e)

def test_chunked_attention_equivalence_cpu():
    """Verify that chunked_scaled_dot_product_attention matches F.scaled_dot_product_attention on CPU."""
    
    # Define test tensor shapes
    batch_size, num_heads, seq_len, head_dim = 2, 4, 128, 32
    key_seq_len = 64
    
    # Create random tensors on CPU
    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cpu')
    key = torch.randn(batch_size, num_heads, key_seq_len, head_dim, device='cpu')
    value = torch.randn(batch_size, num_heads, key_seq_len, head_dim, device='cpu')
    # Create a plausible attention mask with some randomness (e.g., mix of 0s and 1s)
    attention_mask = torch.randint(0, 2, (batch_size, num_heads, seq_len, key_seq_len), device='cpu', dtype=torch.bool)

    # 1. Calculate expected output using the standard PyTorch function
    logger.info("Calculating expected output with F.scaled_dot_product_attention...")
    expected_output = F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )
    logger.info(f"Expected output shape: {expected_output.shape}")

    # 2. Calculate output using the chunked function with a chunk size that forces chunking
    chunk_size = seq_len // 4 # Ensure chunking happens
    logger.info(f"Calculating output with chunked_scaled_dot_product_attention (chunk_size={chunk_size})...")
    chunked_output = chunked_scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, chunk_size=chunk_size
    )
    logger.info(f"Chunked output shape: {chunked_output.shape}")

    # 3. Compare the results
    assert torch.allclose(expected_output, chunked_output, atol=1e-6), \
        "Chunked attention output does not match standard attention output on CPU when chunking."
    logger.info("Equivalence test passed when chunking is forced.")

    # 4. Test the case where chunking should NOT happen (chunk_size >= seq_len)
    large_chunk_size = seq_len * 2
    logger.info(f"Calculating output with chunked_scaled_dot_product_attention (chunk_size={large_chunk_size}, should not chunk)...")
    non_chunked_output = chunked_scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, chunk_size=large_chunk_size
    )
    logger.info(f"Non-chunked output shape: {non_chunked_output.shape}")

    # 5. Compare again
    assert torch.allclose(expected_output, non_chunked_output, atol=1e-6), \
        "Chunked attention output does not match standard attention output on CPU when chunking is skipped."
    logger.info("Equivalence test passed when chunking is skipped.")