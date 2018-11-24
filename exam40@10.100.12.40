#include <iostream>
#include <stdio.h>
#include <sstream>
#include <string.h>
#include <cuda.h>

#define ll long long 

using namespace std;

__global__
void betweenness_centrality_kernel (int nodes, int *C, int *R, int *d, int *sigma, float *delta, float *bc, int *S, int *end_point) {
    
    __shared__ int position;
    __shared__ int s;
    __shared__ int end_pos;
    
    int idx = threadIdx.x;
    if (idx == 0) {
        s = 0;
    }
    __syncthreads();
    
    while (s < nodes) {
        __syncthreads();
        
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
        if(idx == 0) {
            done = false;
            current_depth = 0;
            position = 0;
            end_pos = 1;
            end_point[0] = 0;
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
                    
                    
                    int t = atomicAdd(&position,1);
                    S[t] = v;

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
                end_point[end_pos] = position;
                ++end_pos;
            }
        }


        // Parallel Vertex Parallel implementation
   
        // __syncthreads();
	if(idx == 0){
		end_pos-=2;
		// printf("%d %d %d<--", end_pos, end_point[end_pos], end_point[end_pos+1]);
		for(int a1=0;a1<=end_pos+1;++a1) printf("%d-", end_point[a1]);
		printf("\n");
	    for(int a1=0;a1<nodes;++a1) printf("%d<", S[a1]);
        // cout<<"\n";
        printf("\n");
	} 
        __syncthreads();
	//atomicSub(&end_pos,2);
        for(int itr1 = end_pos; itr1 >= 0; --itr1){
            __syncthreads();
		    for(int itr2 = end_point[itr1] + idx; itr2 < end_point[itr1+1]; itr2+=blockDim.x){
                // S[itr2] is one node
                for(int itr3 = R[S[itr2]]; itr3 < R[S[itr2] + 1]; ++itr3){
                    int consider = C[itr3];
                    // C[itr3] other node
                    if(d[consider] == d[S[itr2]]-1){
                        atomicAdd(&delta[consider], ( ((float)sigma[consider]/sigma[S[itr2]]) * ((float)1 + delta[S[itr2]]) )); 
                    }
                }
                if(S[itr2] != s){
                    bc[S[itr2]] += delta[S[itr2]];
                }

            }
            __syncthreads();
        }

        // Serialized Vertex Parallel implementation

        // if(idx == 0){
            
        //     for(int itr1 = nodes - 1; itr1 >= 0; --itr1){
        //         for(int itr2 = R[S[itr1]]; itr2 < R[S[itr1] + 1]; ++itr2){
        //             int consider = C[itr2];
        //             if(d[consider] == d[S[itr1]]-1){
        //                 delta[consider] += ( ((float)sigma[consider]/sigma[S[itr1]]) * ((float)1 + delta[S[itr1]]) ); 
        //             }
        //         }
        //         if(S[itr1] != s){
        //             bc[S[itr1]] += delta[S[itr1]];
        //         }
        //     }
        // }


        __syncthreads();
        if (idx == 0) {
            s += 1;
        }
       
    }
}



int main () {

    // freopen("graph", "r", stdin);

    // nodes and edges
    int nodes, edges;
    cin>>nodes>>edges;

    // compressed adjancency list
    int * V = new int[nodes + 1];
    int * E = new int[2 * edges];

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

    int *d = new int[nodes];
    int *sigma = new int[nodes];
    float *delta = new float[nodes];
    float *bc = new float[nodes];

    memset(bc,0,sizeof(bc));

    int *d_d, *d_sigma, *d_V, *d_E, *d_S, *d_end_point;
    float *d_delta, *d_bc;

    cudaMalloc((void**)&d_d, sizeof(int) * nodes);
    cudaMalloc((void**)&d_end_point, sizeof(int) * (nodes + 1));
    cudaMalloc((void**)&d_sigma, sizeof(int) * nodes);
    cudaMalloc((void**)&d_S, sizeof(int) * nodes);
    cudaMalloc((void**)&d_V, sizeof(int) * (nodes + 1));
    cudaMalloc((void**)&d_E, sizeof(int) * (2*edges));
    cudaMalloc((void**)&d_delta, sizeof(float) * nodes);
    cudaMalloc((void**)&d_bc, sizeof(float) * nodes);

    cudaMemcpy(d_V, V, sizeof(int) * (nodes+1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_E, E, sizeof(int) * (2*edges), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bc, bc, sizeof(float) * (nodes), cudaMemcpyHostToDevice);    
    // cudaMemcpy(d_delta, delta, sizeof(float) * (nodes), cudaMemcpyHostToDevice);
    
    betweenness_centrality_kernel <<<1, 64>>> (nodes, d_E, d_V, d_d, d_sigma, d_delta, d_bc, d_S, d_end_point);

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


    cudaFree(d_sigma);
    cudaFree(d_d);
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_delta);
    cudaFree(d_bc);
    cudaFree(d_S);
    cudaFree(d_end_point);

    free(E);
    free(V);
    free(d);
    free(sigma);
    free(delta);
    free(bc);

    return 0;
}
