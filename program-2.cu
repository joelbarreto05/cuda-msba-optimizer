#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>

#define N 32 
#define NUM_SWARMS 8
#define BEES_PER_SWARM 256

// Kernel para inicializar estados aleatorios por hilo
__global__ void setup_kernel(curandState *state) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(1234, idx, 0, &state[idx]);
}

template <typename SearchPolicy>
__global__ void msba_kernel(float* d_swarm_positions, const float* d_flow, const float* d_dist, float* d_fitness, curandState *state) {
    __shared__ float s_dist[N][N];
    __shared__ float s_flow[N][N];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    for (int i = tid; i < N * N; i += blockDim.x) {
        ((float*)s_dist)[i] = d_dist[i];
        ((float*)s_flow)[i] = d_flow[i];
    }
    __syncthreads();

    if (idx < NUM_SWARMS * BEES_PER_SWARM) {
        float* bee_pos = &d_swarm_positions[idx * N];
        curandState localState = state[idx];

        // Lógica de Movimiento Estocástico
        for(int iter = 0; iter < 100; ++iter) {
            // Swap aleatorio
            int i = curand(&localState) % N;
            int j = curand(&localState) % N;
            float temp = bee_pos[i];
            bee_pos[i] = bee_pos[j];
            bee_pos[j] = temp;
            
            // Cálculo de Fitness (QAP)
            float fitness = 0.0f;
            for(int k = 0; k < N; ++k) {
                for(int l = 0; l < N; ++l) {
                    fitness += s_flow[k][l] * s_dist[(int)bee_pos[k]][(int)bee_pos[l]];
                }
            }
            d_fitness[idx] = fitness;
        }
        state[idx] = localState;
    }
}

void qap_cpu(const std::vector<float>& flow, const std::vector<float>& dist, int n) {
    float fitness = 0.0f;
    for(int i = 0; i < n * n; i++) fitness += flow[i] * dist[i];
    std::cout << "CPU Base (Costo): " << fitness << std::endl;
}

int main() {
    std::vector<float> h_dist(N * N, 0.1f);
    std::vector<float> h_flow(N * N, 0.2f);
    
    float *d_pos, *d_flow, *d_dist, *d_fitness;
    curandState *d_state;

    cudaMalloc(&d_pos, NUM_SWARMS * BEES_PER_SWARM * N * sizeof(float));
    cudaMalloc(&d_flow, N * N * sizeof(float));
    cudaMalloc(&d_dist, N * N * sizeof(float));
    cudaMalloc(&d_fitness, NUM_SWARMS * BEES_PER_SWARM * sizeof(float));
    cudaMalloc(&d_state, NUM_SWARMS * BEES_PER_SWARM * sizeof(curandState));

    cudaMemcpy(d_flow, h_flow.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, h_dist.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);

    // ... tras los cudaMemcpy ...
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    
    float total_ms = 0;
    for(int i = 0; i < 100; i++) {
        cudaEventRecord(start);
        msba_kernel<int><<<NUM_SWARMS, BEES_PER_SWARM>>>(d_pos, d_flow, d_dist, d_fitness, d_state);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    std::cout << "Promedio GPU (100 iteraciones): " << total_ms / 100.0f << " ms" << std::endl;
    qap_cpu(h_flow, h_dist, N); // Llamada a la referencia CPU
}
    // ... (restos de cudaFree) ...
