# cuda-msba-optimizer

GPU-accelerated implementation of the Multi-Swarm Bees Algorithm (MSBA) for solving Quadratic Assignment Problems (QAP).

## Overview
This project optimizes QAP search processes using a parallelized CUDA architecture. By leveraging thread-level parallelism and shared memory, we achieve significant latency reduction compared to sequential CPU implementations.

## Performance
- **CPU (Baseline):** 20.48 ms
- **GPU (Optimized):** 3.19 ms
- **Speedup:** ~6.4x

## Requirements
- NVIDIA GPU (Compute Capability 7.5+ recommended)
- CUDA Toolkit
- Compiler: `nvcc`

## Compilation & Execution
1. Compile:
   ```bash
   nvcc program.cu -o msba_exec -lcurand
2. Execute:
   ```bash
   ./msba_exec
