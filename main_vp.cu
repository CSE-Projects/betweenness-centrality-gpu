#include <iostream>
#include <stdio.h>
#include <sstream>
#include <string.h>
#include <cuda.h>

#define ll long long 

using namespace std;

__global__
void betweenness_centrality_kernel (int nodes, int *C, int *R, int *d, int *sigma, float *delta, float *bc) {
    // Remove
    __shared__ int S[9];
    __shared__ int position;
    __shared__ int s;
    
    int idx = threadIdx.x;
    if (idx == 0) {
        s = 0;
    }
    __syncthreads();
    
    while (s < nodes) {
        __syncthreads();
        //Initialize d and sigma
        for(int v=idx; v<nodes; v+=blockDim.x) {
            if (idx == 1) {
                printf("Hello %d\n", s);
            }
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
            for (int i = 0; i < nodes; i++) {
                printf("%d-> %d %d\n", i , d[i], s);
            }
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
                    
                    // Remove
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
            }
        }

        // // Remove
        // __syncthreads();
        if(idx == 0){
            // for (int i = 0; i < nodes; i++) {
            //     printf("%d-> %d\n", i , d[i]);
            // }
            for(int itr1 = nodes - 1; itr1 >= 0; --itr1){
                for(int itr2 = R[S[itr1]]; itr2 < R[S[itr1] + 1]; ++itr2){
                    int consider = C[itr2];
                    if(d[consider] == d[S[itr1]]-1){
                        delta[consider] += ( ((float)sigma[consider]/sigma[S[itr1]]) * ((float)1 + delta[S[itr1]]) ); 
                    }
                }
                if(S[itr1] != s){
                    bc[S[itr1]] += 1;//delta[S[itr1]];
                }
            }
        }


        __syncthreads();
        if (idx == 0) {
            s += 1;
        }
        __syncthreads();
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
    
    cout<<"\n";
    for (int i = 0; i <= nodes; i++) {
        cout<<V[i]<<" ";
    }
    cout<<"\n";
    for (int i = 0; i < 2 * edges; ++i) {
        cout<<E[i]<<" ";
    }
    cout<<"\n";

    int *d = new int[nodes];
    int *sigma = new int[nodes];
    float *delta = new float[nodes];
    float *bc = new float[nodes];

    int *d_d, *d_sigma, *d_V, *d_E;
    float *d_delta, *d_bc;

    cudaMalloc((void**)&d_d, sizeof(int) * nodes);
    cudaMalloc((void**)&d_sigma, sizeof(int) * nodes);
    cudaMalloc((void**)&d_V, sizeof(int) * (nodes + 1));
    cudaMalloc((void**)&d_E, sizeof(int) * (2*edges));
    cudaMalloc((void**)&d_delta, sizeof(float) * nodes);
    cudaMalloc((void**)&d_bc, sizeof(float) * nodes);

    cudaMemcpy(d_V, V, sizeof(int) * (nodes+1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_E, E, sizeof(int) * (2*edges), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bc, bc, sizeof(float) * (nodes), cudaMemcpyHostToDevice);    
    cudaMemcpy(d_delta, delta, sizeof(float) * (nodes), cudaMemcpyHostToDevice);
    
    betweenness_centrality_kernel <<<1, 64>>> (nodes, 0, d_E, d_V, d_d, d_sigma, d_delta, d_bc);

    // cudaMemcpy(d, d_d, sizeof(float) * nodes, cudaMemcpyDeviceToHost);
    // cudaMemcpy(sigma, d_sigma, sizeof(float) * nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(bc, d_bc, sizeof(float) * nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(delta, d_delta, sizeof(float) * nodes, cudaMemcpyDeviceToHost);

    cout<<"Res: \n";
    for (int i = 0; i < nodes; i++) {
        printf("%f ", bc[i]);
        // cout<<bc[i];
    }
    cout<<endl;


    cudaFree(d_sigma);
    cudaFree(d_d);
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_delta);
    cudaFree(d_bc);


    return 0;
}

/*
9 14
2 3 4
1 3
1 2 4
1 3 5 6
4 6 7 8
4 5 7 8
5 6 8 9
5 6 7
7
*/