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


// ============== Kernel for betweenness calculation ========================
                
__global__
void betweenness_centrality_kernel (int nodes, int *C, int *R, int *d, int *sigma, float *delta, float *bc, int *reverse_stack) {
    

    // Used to store the position where nodes are pushed as a stack
    __shared__ int position;
    
    // Used to store the source vertex            
    __shared__ int s;
    //__shared__ int end_pos;
    
    int idx = threadIdx.x;
    if (idx == 0) {
        // Initializing source
        s = 0;
        //end_pos = 1;
        //reverse_bfs_limit[0] = 0;
    }
    __syncthreads();
    
    while (s < nodes) {
        __syncthreads();
        
        // ============== Vertex parallel method for BFS ========================
                
        //Initialize d and sigma
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

        // ============== INIT ========================
                
        if(idx == 0) {
            done = false;
            current_depth = 0;
            position = 0;
        }
        __syncthreads();
        
        // SP Calc 
        while(!done)
        {
            __syncthreads();
            done = true;
            __syncthreads();
            
            for(int v=idx; v<nodes; v+=blockDim.x) {
                if(d[v] == current_depth) {
                    
                    // ============== Storing nodes for reverse BFS ========================
                
                    int t = atomicAdd(&position,1);
                    reverse_stack[t] = v;

                    // ============== Relaxation step to find minimum distance ========================
                
                    for(int r=R[v]; r<R[v+1]; r++) {
                        int w = C[r];
                        if(d[w] == INT_MAX) {
                            d[w] = d[v] + 1;
                            done = false;
                        }
                        if(d[w] == (d[v] + 1)) {
                            atomicAdd(&sigma[w],sigma[v]);
                        }
                    }
                }
            }
            __syncthreads();
            if(idx == 0){
                current_depth++;
                //reverse_bfs_limit[end_pos] = position;
                //++end_pos;
            }
        }


        // Parallel Vertex Parallel implementation (uncomment the following lines and comment the ones below)
   
        __syncthreads();
        // atomicSub(&end_pos,2);
        // for(int itr1 = end_pos; itr1 >= 0; --itr1){
        //     for(int itr2 = reverse_bfs_limit[itr1] + idx; itr2 < reverse_bfs_limit[itr1+1]; itr2+=blockDim.x){
        //         // reverse_stack[itr2] is one node
        //         for(int itr3 = R[reverse_stack[itr2]]; itr3 < R[reverse_stack[itr2] + 1]; ++itr3){
        //             int consider = C[itr3];
        //             // C[itr3] other node
        //             if(d[consider] == d[reverse_stack[itr2]]-1){
        //                 delta[consider] += ( ((float)sigma[consider]/sigma[reverse_stack[itr2]]) * ((float)1 + delta[reverse_stack[itr2]]) ); 
        //             }
        //         }
        //         if(reverse_stack[itr2] != s){
        //             bc[reverse_stack[itr2]] += delta[reverse_stack[itr2]];
        //         }

        //     }
        //     __syncthreads();
        // }

        
        // Serialized Vertex Parallel implementation. Comment the following for parallel implementation

        if(idx == 0){
            
            for(int itr1 = nodes - 1; itr1 >= 0; --itr1){
                for(int itr2 = R[reverse_stack[itr1]]; itr2 < R[reverse_stack[itr1] + 1]; ++itr2){
                    int consider = C[itr2];
                    if(d[consider] == d[reverse_stack[itr1]]-1){
                        delta[consider] += ( ((float)sigma[consider]/sigma[reverse_stack[itr1]]) * ((float)1 + delta[reverse_stack[itr1]]) ); 
                    }
                }
                if(reverse_stack[itr1] != s){
                    bc[reverse_stack[itr1]] += delta[reverse_stack[itr1]];
                }
            }
        }

        // ============== Incrementing source ========================
                
        __syncthreads();
        if (idx == 0) {
            s += 1;
        }
       
    }
}



int main () {

    // Uncomment for reading files in stdin
    // freopen("graph", "r", stdin);
    
    // ============== INIT ========================
                
    // nodes and edges
    int nodes, edges;
    cin>>nodes>>edges;

    // compressed adjancency list
    int * V = new int[nodes + 1];
    int * E = new int[2 * edges];

    // ============== Formation of compressed adjacency for CSR ========================
                
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
    
    // Uncomment for printing compressed adjacency list

    // cout<<"\n";
    // for (int i = 0; i <= nodes; i++) {
    //     cout<<V[i]<<" ";
    // }
    // cout<<"\n";
    // for (int i = 0; i < 2 * edges; ++i) {
    //     cout<<E[i]<<" ";
    // }
    // cout<<"\n";

    // Initializations

    int *d = new int[nodes];
    int *sigma = new int[nodes];
    float *delta = new float[nodes];
    float *bc = new float[nodes];

    memset(bc,0,sizeof(bc));

    int *d_d, *d_sigma, *d_V, *d_E, *d_reverse_stack;
    float *d_delta, *d_bc;

    // Allocating memory via cudamalloc

    cudaMalloc((void**)&d_d, sizeof(int) * nodes);
    // cudaMalloc((void**)&d_end_point, sizeof(int) * (nodes + 1));
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
    

    // ============== Kernel call ========================
                

    betweenness_centrality_kernel <<<1, 256>>> (nodes, d_E, d_V, d_d, d_sigma, d_delta, d_bc, d_reverse_stack);

    // cudaMemcpy(d, d_d, sizeof(float) * nodes, cudaMemcpyDeviceToHost);
    // cudaMemcpy(sigma, d_sigma, sizeof(float) * nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(bc, d_bc, sizeof(float) * nodes, cudaMemcpyDeviceToHost);
    // cudaMemcpy(delta, d_delta, sizeof(float) * nodes, cudaMemcpyDeviceToHost);

    cout<<"Res: \n";
    for (int i = 0; i < nodes; i++) {
        printf("%f ", bc[i]/2.0);
        // cout<<bc[i];
    }
    cout<<endl;


    // ============== Deallocating memory ========================

    cudaFree(d_sigma);
    cudaFree(d_d);
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_delta);
    cudaFree(d_bc);
    cudaFree(d_reverse_stack);
    // cudaFree(d_end_point);

    free(E);
    free(V);
    free(d);
    free(sigma);
    free(delta);
    free(bc);

    return 0;
}
