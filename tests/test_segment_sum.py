#!/usr/bin/env python3
"""
JAX Segment Sum Benchmark: FP32 vs BF16 Performance Analysis
"""

import jax
import jax.numpy as jnp
from jax.ops import segment_sum
import numpy as np
import time
from functools import partial

# Ensure JAX uses GPU
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")

# Workload configurations
WORKLOADS = {
    "Workload A": {
        "num_atoms": 43,
        "num_edges": 875,
        "num_lr_pairs": 1723,
        "num_features": 128,
    },
    "Workload B": {
        "num_atoms": 35,
        "num_edges": 841,
        "num_lr_pairs": 1123,
        "num_features": 128,
    },
    "Workload C": {
        "num_atoms": 61,
        "num_edges": 1427,
        "num_lr_pairs": 3541,
        "num_features": 128,
    },
    "Workload D": {
        "num_atoms": 192,
        "num_edges": 7433,
        "num_lr_pairs": 80990208,  
        "num_features": 128,
    },
}

def generate_workload(num_atoms, num_edges, num_lr_pairs, num_features, dtype=jnp.float32):
    """Generate synthetic workload data."""
    key = jax.random.PRNGKey(42)
    
    # Short-range edges 
    key, subkey = jax.random.split(key)
    edge_data = jax.random.normal(subkey, (num_edges, num_features), dtype=dtype)
    
    key, subkey = jax.random.split(key)
    edge_idx_i = jax.random.randint(subkey, (num_edges,), 0, num_atoms)
    
    # Long-range pairs
    key, subkey = jax.random.split(key)
    lr_data = jax.random.normal(subkey, (num_lr_pairs,), dtype=dtype)
    
    key, subkey = jax.random.split(key)
    lr_idx_i = jax.random.randint(subkey, (num_lr_pairs,), 0, num_atoms)
    
    return {
        "edge_data": edge_data,
        "edge_idx_i": edge_idx_i,
        "lr_data": lr_data,
        "lr_idx_i": lr_idx_i,
        "num_atoms": num_atoms,
    }


# ============================================================================
# Benchmark Functions
# ============================================================================

@partial(jax.jit, static_argnums=(2,))
def segment_sum_fp32(data, segment_ids, num_segments):
    """FP32 segment_sum (baseline)."""
    return segment_sum(data, segment_ids=segment_ids, num_segments=num_segments)


@partial(jax.jit, static_argnums=(2,))
def segment_sum_bf16_with_conversion(data, segment_ids, num_segments):
    """BF16 segment_sum with FP32->BF16 conversion included."""
    data_bf16 = data.astype(jnp.bfloat16)
    result_bf16 = segment_sum(data_bf16, segment_ids=segment_ids, num_segments=num_segments)
    return result_bf16.astype(jnp.float32)


@partial(jax.jit, static_argnums=(2,))
def segment_sum_bf16_no_conversion(data_bf16, segment_ids, num_segments):
    """BF16 segment_sum without conversion (data already BF16)."""
    return segment_sum(data_bf16, segment_ids=segment_ids, num_segments=num_segments)


def benchmark_function(fn, data, segment_ids, num_segments, warmup=5, iterations=100, name=""):
    """Benchmark a segment_sum function."""
    # Warmup
    for _ in range(warmup):
        result = fn(data, segment_ids, num_segments)
        result.block_until_ready()
    
    # Timed iterations
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        result = fn(data, segment_ids, num_segments)
        result.block_until_ready()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    times = np.array(times)
    return {
        "name": name,
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "median_ms": np.median(times),
    }


def run_benchmark_suite():
    """Run full benchmark suite."""
    print("\n" + "=" * 80)
    print("JAX SEGMENT_SUM BENCHMARK: FP32 vs BF16")
    print("=" * 80)
    
    results = {}
    
    for workload_name, config in WORKLOADS.items():
        print(f"\n{'=' * 80}")
        print(f"Workload: {workload_name}")
        print(f"  Atoms: {config['num_atoms']}, Edges: {config['num_edges']}, LR Pairs: {config['num_lr_pairs']:,}")
        print("=" * 80)
        
        if config['num_lr_pairs'] > 4000:
            iterations = 10
            warmup = 2
        else:
            iterations = 100
            warmup = 10
        
        # Generate workload data
        workload_fp32 = generate_workload(**config, dtype=jnp.float32)
        workload_bf16 = generate_workload(**config, dtype=jnp.bfloat16)
        
        workload_results = {}
        
        # ====================================================================
        # Test 1: Short-range edges 
        # ====================================================================
        print(f"\n--- Short-Range Edges ({config['num_edges']} edges, {config['num_features']} features) ---")
        
        # FP32
        result = benchmark_function(
            segment_sum_fp32,
            workload_fp32["edge_data"],
            workload_fp32["edge_idx_i"],
            config["num_atoms"],
            warmup=warmup,
            iterations=iterations,
            name="FP32"
        )
        print(f"  FP32:                    {result['mean_ms']:.4f} ± {result['std_ms']:.4f} ms")
        workload_results["edge_fp32"] = result
        
        # BF16 with conversion
        result = benchmark_function(
            segment_sum_bf16_with_conversion,
            workload_fp32["edge_data"],
            workload_fp32["edge_idx_i"],
            config["num_atoms"],
            warmup=warmup,
            iterations=iterations,
            name="BF16 (with conversion)"
        )
        print(f"  BF16 (with conversion):  {result['mean_ms']:.4f} ± {result['std_ms']:.4f} ms")
        workload_results["edge_bf16_conv"] = result
        
        # BF16 without conversion
        result = benchmark_function(
            segment_sum_bf16_no_conversion,
            workload_bf16["edge_data"],
            workload_bf16["edge_idx_i"],
            config["num_atoms"],
            warmup=warmup,
            iterations=iterations,
            name="BF16 (no conversion)"
        )
        print(f"  BF16 (no conversion):    {result['mean_ms']:.4f} ± {result['std_ms']:.4f} ms")
        workload_results["edge_bf16_no_conv"] = result
        
        # Speedup analysis
        fp32_time = workload_results["edge_fp32"]["mean_ms"]
        bf16_conv_time = workload_results["edge_bf16_conv"]["mean_ms"]
        bf16_no_conv_time = workload_results["edge_bf16_no_conv"]["mean_ms"]
        
        print(f"\n  Speedup Analysis (edges):")
        print(f"    BF16 (with conv) vs FP32: {fp32_time/bf16_conv_time:.2f}x {'faster' if bf16_conv_time < fp32_time else 'SLOWER'}")
        print(f"    BF16 (no conv) vs FP32:   {fp32_time/bf16_no_conv_time:.2f}x {'faster' if bf16_no_conv_time < fp32_time else 'SLOWER'}")
        
        # ====================================================================
        # Test 2: Long-range pairs 
        # ====================================================================
        print(f"\n--- Long-Range Pairs ({config['num_lr_pairs']:,} pairs, scalar) ---")
        
        # FP32
        result = benchmark_function(
            segment_sum_fp32,
            workload_fp32["lr_data"],
            workload_fp32["lr_idx_i"],
            config["num_atoms"],
            warmup=warmup,
            iterations=iterations,
            name="FP32"
        )
        print(f"  FP32:                    {result['mean_ms']:.4f} ± {result['std_ms']:.4f} ms")
        workload_results["lr_fp32"] = result
        
        # BF16 with conversion
        result = benchmark_function(
            segment_sum_bf16_with_conversion,
            workload_fp32["lr_data"],
            workload_fp32["lr_idx_i"],
            config["num_atoms"],
            warmup=warmup,
            iterations=iterations,
            name="BF16 (with conversion)"
        )
        print(f"  BF16 (with conversion):  {result['mean_ms']:.4f} ± {result['std_ms']:.4f} ms")
        workload_results["lr_bf16_conv"] = result
        
        # BF16 without conversion
        result = benchmark_function(
            segment_sum_bf16_no_conversion,
            workload_bf16["lr_data"],
            workload_bf16["lr_idx_i"],
            config["num_atoms"],
            warmup=warmup,
            iterations=iterations,
            name="BF16 (no conversion)"
        )
        print(f"  BF16 (no conversion):    {result['mean_ms']:.4f} ± {result['std_ms']:.4f} ms")
        workload_results["lr_bf16_no_conv"] = result
        
        # Speedup analysis
        fp32_time = workload_results["lr_fp32"]["mean_ms"]
        bf16_conv_time = workload_results["lr_bf16_conv"]["mean_ms"]
        bf16_no_conv_time = workload_results["lr_bf16_no_conv"]["mean_ms"]
        
        print(f"\n  Speedup Analysis (LR pairs):")
        print(f"    BF16 (with conv) vs FP32: {fp32_time/bf16_conv_time:.2f}x {'faster' if bf16_conv_time < fp32_time else 'SLOWER'}")
        print(f"    BF16 (no conv) vs FP32:   {fp32_time/bf16_no_conv_time:.2f}x {'faster' if bf16_no_conv_time < fp32_time else 'SLOWER'}")
        
        results[workload_name] = workload_results
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\n{:<30} {:>12} {:>12} {:>12} {:>12}".format(
        "Workload", "FP32 (ms)", "BF16+conv", "BF16 only", "Speedup"
    ))
    print("-" * 80)
    
    for workload_name, workload_results in results.items():
        # Long-range pairs
        fp32 = workload_results["lr_fp32"]["mean_ms"]
        bf16_conv = workload_results["lr_bf16_conv"]["mean_ms"]
        bf16_no_conv = workload_results["lr_bf16_no_conv"]["mean_ms"]
        speedup = fp32 / bf16_no_conv
        
        print("{:<30} {:>12.4f} {:>12.4f} {:>12.4f} {:>11.2f}x".format(
            workload_name[:30], fp32, bf16_conv, bf16_no_conv, speedup
        ))

if __name__ == "__main__":
    run_benchmark_suite()
