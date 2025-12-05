#include <math.h>
#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"

// We use the existing global host arrays declared in vector.h:
// extern vector3 *hVel;
// extern vector3 *hPos;
// extern double *mass;

// Kernel 1: compute pairwise accelerations: effect of j on i
__global__
void compute_accels_kernel(const vector3 *pos,
                           const double *mass,
                           vector3 *accels,
                           int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (i >= n || j >= n) return;

    vector3 a;

    if (i == j) {
        a[0] = a[1] = a[2] = 0.0;
    } else {
        double dx = pos[i][0] - pos[j][0];
        double dy = pos[i][1] - pos[j][1];
        double dz = pos[i][2] - pos[j][2];

        double magnitude_sq = dx * dx + dy * dy + dz * dz + 1e-12; // avoid div by 0
        double magnitude     = sqrt(magnitude_sq);
        double accelmag      = -1.0 * GRAV_CONSTANT * mass[j] / magnitude_sq;

        a[0] = accelmag * dx / magnitude;
        a[1] = accelmag * dy / magnitude;
        a[2] = accelmag * dz / magnitude;
    }

    // Flattened 2D index: [i][j] -> i * n + j
    accels[i * n + j][0] = a[0];
    accels[i * n + j][1] = a[1];
    accels[i * n + j][2] = a[2];
}

// Kernel 2: for each object i, sum all accels[i][j] over j, then update vel & pos
__global__
void sum_and_update_kernel(vector3 *pos,
                           vector3 *vel,
                           const vector3 *accels,
                           int n,
                           double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;

    // Sum row i across all j
    for (int j = 0; j < n; ++j) {
        int idx = i * n + j;
        ax += accels[idx][0];
        ay += accels[idx][1];
        az += accels[idx][2];
    }

    // Update velocity
    vel[i][0] += ax * dt;
    vel[i][1] += ay * dt;
    vel[i][2] += az * dt;

    // Update position
    pos[i][0] += vel[i][0] * dt;
    pos[i][1] += vel[i][1] * dt;
    pos[i][2] += vel[i][2] * dt;
}

// GPU-based compute() replacing the original CPU version
void compute()
{
    static int initialized = 0;
    static vector3 *dPos   = NULL;
    static vector3 *dVel   = NULL;
    static double  *dMass  = NULL;
    static vector3 *dAccels = NULL;

    const int n = NUMENTITIES;
    const size_t vecSize   = n * sizeof(vector3);
    const size_t massSize  = n * sizeof(double);
    const size_t accelSize = (size_t)n * (size_t)n * sizeof(vector3);

    // One-time device allocation and initial host->device copy
    if (!initialized) {
        cudaMalloc(&dPos, vecSize);
        cudaMalloc(&dVel, vecSize);
        cudaMalloc(&dMass, massSize);
        cudaMalloc(&dAccels, accelSize);

        cudaMemcpy(dPos,  hPos,  vecSize,  cudaMemcpyHostToDevice);
        cudaMemcpy(dVel,  hVel,  vecSize,  cudaMemcpyHostToDevice);
        cudaMemcpy(dMass, mass,  massSize, cudaMemcpyHostToDevice);

        initialized = 1;
    }

    // --- Launch Kernel 1: compute pairwise accelerations ---
    dim3 block2D(16, 16);
    dim3 grid2D((n + block2D.x - 1) / block2D.x,
                (n + block2D.y - 1) / block2D.y);

    compute_accels_kernel<<<grid2D, block2D>>>(dPos, dMass, dAccels, n);

    // --- Launch Kernel 2: sum rows and update vel/pos ---
    int block1D = 256;
    int grid1D  = (n + block1D - 1) / block1D;

    sum_and_update_kernel<<<grid1D, block1D>>>(dPos, dVel, dAccels, n, INTERVAL);

    cudaDeviceSynchronize();

    // Copy updated positions and velocities back to host so printSystem sees them
    cudaMemcpy(hPos, dPos, vecSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, dVel, vecSize, cudaMemcpyDeviceToHost);
}
