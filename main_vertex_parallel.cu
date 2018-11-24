/*
Authors
- Dibyadarshan Hota 16CO154
- Omkar Prabhu 16CO233
*/
#include <iostream>
#include <stdio.h>
#include <sstream>
#include <string.h>
#include <cuda.h>

#define ll long long 

using namespace std;

/**
 * Kernel for computing Betweenness Centrality
 * res: Stored in global memory variable bc
 */
__global__
void betweenness_centrality_kernel (int nodes, int *C, int *R, int *d, int *sigma, float *delta, float *bc, int *reverse_stack, int *finish_limit) {
    // ================================== VARIABLES INIT ============================================
    // initial variables
    __shared__ int position;
    __shared__ int s;
    __shared__ int finish_limit_position;
    
    // source variable initially 0
    int idx = threadIdx.x;
    if (idx == 0) {
        s = 0;
    }
    __syncthreads();
    
    // move through all nodes
    while (s < nodes) {
        __syncthreads();
        
        // ============================== distance, delta and sigma INIT ============================================
        for(int v=idx; v<nodes; v+=blockDim.x) {
            if(v == s) {
                d[v] = 0;
                sigma[v] = 1;
            }
            else {
                d[v] = INT_MAX;
                sigma[v] = 0;
            }
            delta[v] = 0;
        }
        __syncthreads();
        __shared__ int current_depth;
        __shared__ bool done;
        if(idx == 0) {
            done = false;
            current_depth = 0;
            position = 0;
            finish_limit_position = 1;
            finish_limit[0] = 0;
        }
        __syncthreads();
        
        // ============================== Shortest Path Calculation using curr source ======================
        // ============================== Using Vertex Parallel ============================================  
        while(!done)
        {   // wait
            __syncthreads();
            done = true;
            __syncthreads();
            // move through modes
            for(int v=idx; v<nodes; v+=blockDim.x) {
                if(d[v] == current_depth) {
                    // add to reverse_stack 
                    int t = atomicAdd(&position,1);
                    reverse_stack[t] = v;
                    // move through neighbours
                    for(int r=R[v]; r<R[v+1]; r++) {
                        int w = C[r];
                        // if not visited
                        if(d[w] == INT_MAX) {
                            d[w] = d[v] + 1;
                            done = false;
                        }
                        // add number of paths
                        if(d[w] == (d[v] + 1)) {
                            atomicAdd(&sigma[w],sigma[v]);
                        }
                    }
                }
            }
            __syncthreads();
            // increment variables
            if(idx == 0){
                current_depth++;
                finish_limit[finish_limit_position] = position;
                ++finish_limit_position;
            }
        }


        // ============================== BC calculation using Brande's Algorithm ============================================

        // Parallel Vertex Parallel implementation   
        // __syncthreads();
        if(idx == 0){
            finish_limit_position-=2;
            // printf("%d %d %d<--", finish_limit_position, finish_limit[finish_limit_position], finish_limit[finish_limit_position+1]);
            // for(int a1=0;a1<=finish_limit_position+1;++a1) printf("%d-", finish_limit[a1]);
            // printf("\n");
            // for(int a1=0;a1<nodes;++a1) printf("%d<", reverse_stack[a1]);
            // cout<<"\n";
            // printf("\n");
        } 
        __syncthreads();
	    //atomicSub(&finish_limit_position,2);
        for(int itr1 = finish_limit_position; itr1 >= 0; --itr1){
            // __syncthreads();
		    for(int itr2 = finish_limit[itr1] + idx; itr2 < finish_limit[itr1+1]; itr2+=blockDim.x){
                // reverse_stack[itr2] is one node
                for(int itr3 = R[reverse_stack[itr2]]; itr3 < R[reverse_stack[itr2] + 1]; ++itr3){
                    int consider = C[itr3];
                    // C[itr3] other node
                    if(d[consider] == d[reverse_stack[itr2]]+1){
                        //atomicAdd(&delta[consider], ( ((float)sigma[consider]/sigma[reverse_stack[itr2]]) * ((float)1 + delta[reverse_stack[itr2]]) )); 
                        delta[reverse_stack[itr2]] += ( ((float)sigma[reverse_stack[itr2]]/sigma[consider]) * ((float)1 + delta[consider]) );
                    }
                }
                if(reverse_stack[itr2] != s){
                    bc[reverse_stack[itr2]] += delta[reverse_stack[itr2]];
                }

            }
            __syncthreads();
        }

        // Serialized Vertex Parallel implementation

        // if(idx == 0){
            
        //     for(int itr1 = nodes - 1; itr1 >= 0; --itr1){
        //         for(int itr2 = R[reverse_stack[itr1]]; itr2 < R[reverse_stack[itr1] + 1]; ++itr2){
        //             int consider = C[itr2];
        //             if(d[consider] == d[reverse_stack[itr1]]-1){
        //                 delta[consider] += ( ((float)sigma[consider]/sigma[reverse_stack[itr1]]) * ((float)1 + delta[reverse_stack[itr1]]) ); 
        //             }
        //         }
        //         if(reverse_stack[itr1] != s){
        //             bc[reverse_stack[itr1]] += delta[reverse_stack[itr1]];
        //         }
        //     }
        // }

        // increment 
        __syncthreads();
        if (idx == 0) {
            s += 1;
        }
       
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
        V[node] = counter;
        istringstream is(line);
        int tmp;
        while (is >> tmp) {
            E[counter] = tmp;
            counter += 1;
        }
        ++node;
    }
    V[node] = counter;
    
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
    int *d = new int[nodes];
    int *sigma = new int[nodes];
    float *delta = new float[nodes];
    float *bc = new float[nodes];

    memset(bc,0,sizeof(bc));

    int *d_d, *d_sigma, *d_V, *d_E, *d_reverse_stack, *d_end_point;
    float *d_delta, *d_bc;

    cudaMalloc((void**)&d_d, sizeof(int) * nodes);
    cudaMalloc((void**)&d_end_point, sizeof(int) * (nodes + 1));
    cudaMalloc((void**)&d_sigma, sizeof(int) * nodes);
    cudaMalloc((void**)&d_reverse_stack, sizeof(int) * nodes);
    cudaMalloc((void**)&d_V, sizeof(int) * (nodes + 1));
    cudaMalloc((void**)&d_E, sizeof(int) * (2*edges));
    cudaMalloc((void**)&d_delta, sizeof(float) * nodes);
    cudaMalloc((void**)&d_bc, sizeof(float) * nodes);

    cudaMemcpy(d_V, V, sizeof(int) * (nodes+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, E, sizeof(int) * (2*edges), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bc, bc, sizeof(float) * (nodes), cudaMemcpyHostToDevice);    
    // cudaMemcpy(d_delta, delta, sizeof(float) * (nodes), cudaMemcpyHostToDevice);
    
    // ================================ KERNEL PARAMS AND CALL ====================================

    float elapsed_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // kernel call
    betweenness_centrality_kernel <<<1, 1024>>> (nodes, d_E, d_V, d_d, d_sigma, d_delta, d_bc, d_reverse_stack, d_end_point);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    // ================================ RESULT ====================================

    // cudaMemcpy(d, d_d, sizeof(float) * nodes, cudaMemcpyDeviceToHost);
    // cudaMemcpy(sigma, d_sigma, sizeof(float) * nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(bc, d_bc, sizeof(float) * nodes, cudaMemcpyDeviceToHost);
    // cudaMemcpy(delta, d_delta, sizeof(float) * nodes, cudaMemcpyDeviceToHost);

    cout<<"Result: \n";
    // for (int i = 0; i < nodes; i++) {
    //     cout<<"Node: "<<i<<" BC: "<<fixed<<setprecision(6)<<bc[i]/2.0<<"\n";
    // }
    cout<<"\n";
    // Print the time for execution
    cout<<"Execution time: "<<elapsed_time/1000.0<<endl;

    // Maximum BC value
    float max_bc = 0.0;
    for (int i = 0; i < nodes; ++i) {
        max_bc = (bc[i] > max_bc) ? bc[i] : max_bc;
    }
    cout<<"Max BC value: "<<max_bc/2.0<<endl;

    // ================================ MEMORY RELEASE ====================================
    cudaFree(d_sigma);
    cudaFree(d_d);
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_delta);
    cudaFree(d_bc);
    cudaFree(d_reverse_stack);
    cudaFree(d_end_point);

    free(E);
    free(V);
    free(d);
    free(sigma);
    free(delta);
    free(bc);

    return 0;
}
