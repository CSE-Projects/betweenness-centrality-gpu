#include <iostream>
#include <string>
#include <sstream>
#include <cuda.h>
#include<stdio.h>
#include <ctime>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/device_free.h>

#define ll long long 

using namespace std;

/**
 * Kernel for computing Betweenness Centrality
 * res: Stored in global memory variable bc
 */
__global__
void betweenness_centrality_kernel (float *bc, int nodes, int edges, const int *V, const int *E, int *Queue_curr, int *Queue_next, int *Depth_Nodes, int *Depth_Points, unsigned long long *sigma, int *distance, float *delta, int *next_source, size_t scale_distance, size_t scale_sigma, size_t scale_delta, size_t scale_Queue_curr, size_t scale_Queue_next, size_t scale_depthnodes, size_t scale_depthpoints) {

    // ================================== VARIABLES INIT ============================================

    // common global delta, sigma and  distance pointers for
    // offset by blockId times scale for each block to get its storing space
    int *s_distance = (int*)((char*)distance + blockIdx.x*scale_distance);
    unsigned long long *s_sigma = (unsigned long long*)((char*)sigma + blockIdx.x*scale_sigma);
    float *s_delta = (float*)((char*)delta + blockIdx.x*scale_delta);
    // printf ("Thread number %d\n", threadIdx.x);

    // shared pointers local to block
    // pointing to global memory as entire shared memory space is not enough to hold for all blocks
    __shared__ int *s_queue_curr;
    __shared__ int *s_queue_next;
    __shared__ int *s_depthnodes;
    __shared__ int *s_depthpoints;
    __shared__ int curr_source;
    __shared__ int block_source;
    // lengths
    __shared__ int len_queue_curr;
    __shared__ int len_queue_next;
    __shared__ int len_depthnodes;
    __shared__ int len_depthpoints;
    __shared__ int depth; 

    // current thread
    int tid = threadIdx.x;

    // init
    if (tid == 0) {
        // block source
        block_source = blockIdx.x;
        // current source operation
        curr_source = block_source;
        // offset by blockId time scale for getting the space for each block
        s_queue_curr = (int*)((char*)Queue_curr + blockIdx.x*scale_Queue_curr);
        s_queue_next = (int*)((char*)Queue_next + blockIdx.x*scale_Queue_next);
        s_depthnodes = (int*)((char*)Depth_Nodes + blockIdx.x*scale_depthnodes);
        s_depthpoints = (int*)((char*)Depth_Points + blockIdx.x*scale_depthpoints);
        // Check point
        // for (int i = 0; i < nodes; i++) {
        //     printf("%d: %d", i, s_distance[i]);
        // }
        // printf("Block %d I %d", blockIdx, );
    }
    // wait for init to complete
    __syncthreads();

    // ================================== MAIN ============================================
    // check if the block is operating on a valid source i.e < total nodes
    while (block_source < nodes) {
        // ============================== distance, delta and sigma INIT ============================================
        // In parallel
        for(int k = threadIdx.x; k < nodes; k += blockDim.x) {
            if(k == curr_source) {
                s_distance[k] = 0;
                s_sigma[k] = 1;
            }
            else {
                s_distance[k] = INT_MAX;
                s_sigma[k] = 0;
            }	
            s_delta[k] = 0;
        }
        // wait for completion
        __syncthreads();
        
        // ============================== Shortest Path Calculation using curr source ======================
        // ============================== Using Work Efficiency ============================================ 
        
        // init lenghts
        if(tid == 0) {
            s_queue_curr[0] = curr_source;
            len_queue_curr = 1;
            len_queue_next = 0;
            s_depthnodes[0] = curr_source;
            len_depthnodes = 1;
            s_depthpoints[0] = 0;
            s_depthpoints[1] = 1;
            len_depthpoints = 2;
            depth = 0;
            // Check point
            // printf("Block: %d Root: %d\n", blockIdx.x, curr_source);
        }
        __syncthreads();

        // start
        while(1) {
            // In parallel for current queue elements
            for(int k = threadIdx.x; k < len_queue_curr; k += blockDim.x) {
                // get vertex at current depth
                int v = s_queue_curr[k];
                // traverse neighbours
                for(int r = V[v]; r < V[v+1]; r++) {
                    int w = E[r];
                    // update if not already updated
                    if(atomicCAS(&s_distance[w], INT_MAX, s_distance[v]+1) == INT_MAX) {
                        int ii = atomicAdd(&len_queue_next, 1);
                        s_queue_next[ii] = w;
                    }
                    // add the total paths possible to sigma
                    if(s_distance[w] == (s_distance[v]+1)) {
                        atomicAdd(&s_sigma[w], s_sigma[v]);
                    }
                }
            }
            __syncthreads();

            // check if completely traversed
            if(len_queue_next == 0) {
                break;
            }
            else {
                // move next traversal elements from Queue next to Queue curr
                for(int i = threadIdx.x; i < len_queue_next; i += blockDim.x) {
                    s_queue_curr[i] = s_queue_next[i];
                    s_depthnodes[i+len_depthnodes] = s_queue_next[i];
                }
                __syncthreads();
                // Set variables
                if(tid == 0) {
                    s_depthpoints[len_depthpoints] = s_depthpoints[len_depthpoints-1] + len_queue_next;
                    len_depthpoints++;
                    len_queue_curr = len_queue_next;
                    len_depthnodes += len_queue_next;
                    len_queue_next = 0;
                    depth++;
                }
                __syncthreads();
            }
        }

        // get depth for traversal
        if(tid == 0) {
            depth = s_distance[s_depthnodes[len_depthnodes-1]] - 1;
        }
        __syncthreads();
        
        // ============================== BC calculation using Brande's Algorithm ============================================
        // In parallel 
        // go from depth
        while(depth > 0) {
            for(int k = threadIdx.x + s_depthpoints[depth]; k < s_depthpoints[depth+1]; k += blockDim.x) {
                // get elements at this depth
                int w = s_depthnodes[k];
                // init
                float dsw = 0;
                float sw = (float)s_sigma[w];
                // check neighbours
                for(int r = V[w]; r < V[w+1]; r++) {
                    int v = E[r];
                    // neighbour within 1 distance
                    if(s_distance[v] == (s_distance[w]+1)) {
                        // accumulate sigma
                        dsw += (sw/(float)s_sigma[v])*(1.0f+s_delta[v]);
                    }
                }
                // update delta using dsw
                s_delta[w] = dsw;
            }
            // move to higher depth
            __syncthreads();
            if(tid == 0) {
                depth--;
            }
            __syncthreads();
        }
        
        for(int k = threadIdx.x; k < nodes; k += blockDim.x) {
            atomicAdd(&bc[k], s_delta[k]);
        }
        __syncthreads();

        // ============================== NEXT OPERATING SOURCE FOR BLOCK ============================================
        // get a block's next operating source
        // using number of blocks launched stored in next source variable
        if(tid == 0) {
            block_source = atomicAdd(next_source, 1);
            curr_source = block_source;
        }
        __syncthreads();
    }
}


/**
 * Main function
 */
int main () {

    // ================================ READ INPUT AND MAKE Compressed Adjancency List ====================================
    // freopen("graph", "r", stdin);
    
    // nodes and edges
    int nodes, edges;
    cin>>nodes>>edges;

    // compressed adjancency list
    int * V = new int[nodes + 1];
    int * E = new int[2 * edges];

    // read graph data in CSR format 
    string line;
    int node = 0;
    int counter = 0;
    getline(cin, line);
    for (int i = 0; i < nodes; ++i) {
        getline(cin, line);
        // cout<<"->>>"<<node<<" "<<counter<<"\n";
        V[node] = counter;
        istringstream is(line);
        int tmp;
        while (is >> tmp) {
            E[counter] = tmp;
            counter += 1;
            // cout<<"-->"<<node<<" "<<counter<<"\n";
        }
        ++node;
    }
    V[node] = counter;
    // Check compressed adj list value
    // cout<<"\n";
    // for (int i = 0; i <= nodes; i++) {
    //     cout<<V[i]<<" ";
    // }
    // cout<<"\n";
    // for (int i = 0; i < 2 * edges; ++i) {
    //     cout<<E[i]<<" ";
    // }
    // cout<<"\n";


    // ================================ DECLARE AND INIT VARIABLES ====================================
    // host pointer bc holds the final result
    float *bc = new float[nodes];
    // device pointers explained in kernel
    float *d_bc, *d_delta;
    int *d_V, *d_E, *d_distance, *d_Queue1, *d_Queue2, *d_Depth_Nodes, *d_Depth_Points, *d_next_source;
    unsigned long long *d_sigma;
    // offsets filled later
    size_t scale_distance, scale_sigma, scale_delta, scale_Q1, scale_Q2, scale_depthnodes, scale_depthpoints;

    // to get next source vertex after a block finishes computing the BC using the current source 
    int *next_source = new int;
    // set to number of SM as each block is assigned to 1 SM
    // next_source[0] = 5;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop,0);
    next_source[0] = prop.multiProcessorCount;
    
    // Allocate space on device
    cudaMalloc((void**)&d_bc, sizeof(float) * nodes);
    cudaMalloc((void**)&d_V, sizeof(int) * (nodes + 1));
    cudaMalloc((void**)&d_E, sizeof(int) * (2*edges));
    // Allocate these common variables space equal to n times the number of blocks
    // As shared memory won't have enough space to use them for each block
    cudaMallocPitch((void**)&d_Queue1, &scale_Q1, sizeof(int) * nodes, next_source[0]);
    cudaMallocPitch((void**)&d_Queue2, &scale_Q2,sizeof(int) * nodes, next_source[0]);
    cudaMallocPitch((void**)&d_Depth_Nodes, &scale_depthnodes,sizeof(int) * nodes, next_source[0]);
    cudaMallocPitch((void**)&d_Depth_Points, &scale_depthpoints,sizeof(int) * (nodes + 1), next_source[0]);
    cudaMallocPitch((void**)&d_sigma, &scale_sigma, sizeof(unsigned long long) * nodes, next_source[0]);
    cudaMallocPitch((void**)&d_distance, &scale_distance, sizeof(int) * nodes, next_source[0]);
    cudaMallocPitch((void**)&d_delta, &scale_delta, sizeof(float) * nodes, next_source[0]);
    
    cudaMalloc((void**)&d_next_source, sizeof(int));
    
    // Copy Required items from host to device
    cudaMemcpy(d_V, V, sizeof(int) * (nodes+1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_E, E, sizeof(int) * (2*edges), cudaMemcpyHostToDevice);
	cudaMemset(d_bc, 0, sizeof(float) * nodes);
	cudaMemcpy(d_next_source , next_source, sizeof(int), cudaMemcpyHostToDevice);

    // ================================ KERNEL PARAMS AND CALL ====================================
    // Grid parameters
	dim3 cudaGrid, cudaBlock;
	cudaGrid.x = next_source[0];cudaGrid.y = 1;cudaGrid.z = 1;
    // Block parameters
	cudaBlock.x = 64;cudaBlock.y = 1;cudaBlock.z = 1;
    
    float elapsed_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // KERNEL CALL
    betweenness_centrality_kernel <<<cudaGrid, cudaBlock>>> (d_bc, nodes, edges, d_V, d_E, d_Queue1, d_Queue2, d_Depth_Nodes, d_Depth_Points, d_sigma, d_distance, d_delta, d_next_source, scale_distance, scale_sigma, scale_delta, scale_Q1, scale_Q2, scale_depthnodes, scale_depthpoints);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    // ================================ RESULT ====================================
    cudaMemcpy(bc, d_bc, sizeof(float) * nodes, cudaMemcpyDeviceToHost);
    
    cout<<"Result: \n";
    // for (int i = 0; i < nodes; i++) {
    //     cout<<"Node: "<<i<<" BC: "<<fixed<<setprecision(6)<<bc[i]/2.0<<"\n";
    // }
    cout<<"\n";
    // Print the time for execution
    cout<<"Execution time: "<<elapsed_time/1000.0<<endl;

    // Maximum element using thrust min reduction
    // thrust::device_vector<float> device_bc_max(bc, bc + nodes);
    // thrust::device_ptr<float> ptr = device_bc_max.data();
    // int max_index = thrust::max_element(ptr, ptr + nodes) - ptr;
    // cout<<"Max BC value: "<<device_bc_max[max_index]/2.0<<endl;
    // Maximum BC value
    float max_bc = 0.0;
    for (int i = 0; i < nodes; ++i) {
        max_bc = (bc[i] > max_bc) ? bc[i] : max_bc;
    }
    cout<<"Max BC value: "<<max_bc/2.0<<endl;

    // ================================ MEMORY RELEASE ====================================
    // free device variable
    cudaFree(d_bc);
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_Queue1);
    cudaFree(d_Queue2);
    cudaFree(d_Depth_Nodes);
    cudaFree(d_Depth_Points);
    cudaFree(d_sigma);
    cudaFree(d_distance);
    cudaFree(d_delta);
    cudaFree(d_next_source);
    // thrust::device_free(device_bc_max);
    // free host variables
    free(V);
    free(E);
    free(bc);

    return 0;
}
