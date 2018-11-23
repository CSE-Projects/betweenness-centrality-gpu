#include <iostream>
#include <string>
#include <sstream>
#include <cuda.h>

#define ll long long 

using namespace std;

__global__
void betweenness_centrality_kernel (float *bc, int nodes, int edges, const int *R, const int *C, int *Q, int *Q2, int *S, int *endpoints, unsigned long long *sigma, int *d, float *delta, int *next_source) {

    // why not shared
    unsigned long long *sigma_row = (unsigned long long*)((char*)sigma + blockIdx.x * (sizeof(unsigned long long)*nodes * gridDim.x));
    float *delta_row = (float*)((char*)delta + blockIdx.x * (sizeof(float)*nodes * gridDim.x));
    int *d_row = (int*)((char*)d + blockIdx.x * (sizeof(int)*nodes * gridDim.x));

    __shared__ int *Q_row;
	__shared__ int *Q2_row;
	__shared__ int *S_row;
    __shared__ int *endpoints_row;
    __shared__ int i;
    __shared__ int ind;

    int j = threadIdx.x;

    if (j == 0) {
        ind = blockIdx.x;
        i = ind;
        Q_row = (int*)((char*)Q + blockIdx.x * (sizeof(int)*nodes * gridDim.x));
        Q2_row = (int*)((char*)Q2 + blockIdx.x * (sizeof(int)*nodes * gridDim.x));
        S_row = (int*)((char*)S + blockIdx.x * (sizeof(int)*nodes * gridDim.x));
        endpoints_row = (int*)((char*)endpoints + blockIdx.x * (sizeof(int)*nodes * gridDim.x));
    }
    __syncthreads();

    while (ind < nodes) {
        //Initialization
        for(int k = threadIdx.x; k < nodes; k += blockDim.x)
        {
            if(k == i) {
                d_row[k] = 0;
                sigma_row[k] = 1;
            }
            else {
                d_row[k] = INT_MAX;
                sigma_row[k] = 0;
            }	
            delta_row[k] = 0;
        }
        
        //Shortest Path Calculation
        __shared__ int Q_len;
        __shared__ int Q2_len;
        __shared__ int S_len;
        __shared__ int current_depth; 
        __shared__ int endpoints_len;

        if(j == 0)
        {
            Q_row[0] = i;
            Q_len = 1;
            Q2_len = 0;
            S_row[0] = i;
            S_len = 1;
            endpoints_row[0] = 0;
            endpoints_row[1] = 1;
            endpoints_len = 2;
            current_depth = 0;
        }
        __syncthreads();

        while (1) {            
            __shared__ int next_index;
            if(j == 0) {
                next_index = blockDim.x;
            }
            __syncthreads();
            
            int k = threadIdx.x;
            while(k < Q_len) {
                int v = Q_row[k];
                for(int r = R[v]; r < R[v+1]; r++) {
                    int w = C[r];
                    
                    if(atomicCAS(&d_row[w], INT_MAX, d_row[v]+1) == INT_MAX) {
                        int t = atomicAdd(&Q2_len, 1);
                        Q2_row[t] = w;
                    }
                    if(d_row[w] == (d_row[v]+1)) {
                        atomicAdd(&sigma_row[w], sigma_row[v]);
                    }
                }
                k = atomicAdd(&next_index, 1);
            }
            __syncthreads();

			if(Q2_len == 0) {
				break;
			}
			else {
				for(int kk = threadIdx.x; kk < Q2_len; kk += blockDim.x) {
					Q_row[kk] = Q2_row[kk];
					S_row[kk+S_len] = Q2_row[kk];
				}
                __syncthreads();
				if(j == 0) {
					endpoints_row[endpoints_len] = endpoints_row[endpoints_len-1] + Q2_len;
					endpoints_len++;
					Q_len = Q2_len;
					S_len += Q2_len;
					Q2_len = 0;
					// current_depth++;
				}
				__syncthreads();
			}
        }

        if(j == 0) {
			current_depth = d_row[S_row[S_len-1]] - 1;
		}
        __syncthreads();
        
        //Dependency Accumulation
		while(current_depth > 0) {
            for(int kk=threadIdx.x + endpoints_row[current_depth]; kk < endpoints_row[current_depth+1]; kk += blockDim.x)
            {
                int w = S_row[kk];
                float dsw = 0;
                float sw = (float)sigma_row[w];
                for(int z = R[w]; z < R[w+1]; z++)
                {
                    int v = C[z];
                    if(d_row[v] == (d_row[w]+1))
                    {
                        dsw += (sw/(float)sigma_row[v])*(1.0f+delta_row[v]);
                    }
                }
                delta_row[w] = dsw;
            }
			__syncthreads();
			if(j == 0)
			{
				current_depth--;
			}
			__syncthreads();
        }
        
        for(int kk = threadIdx.x; kk < nodes; kk += blockDim.x) {
			atomicAdd(&bc[kk], delta_row[kk]);
		}
		
		if(j == 0) {
			ind = atomicAdd(next_source, 1);
            i = ind;
		}
		__syncthreads();
    }
}


int main () {

    freopen("graph", "r", stdin);
    
    // nodes and edges
    int nodes, edges;
    cin>>nodes>>edges;

    // compressed adjancency list
    int * V = new int[nodes];
    int * E = new int[2 * edges + 1];

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
    V[node + 1] = 2 * edges;
    
    cout<<"\n";
    for (int i = 0; i < nodes; i++) {
        cout<<V[i]<<" ";
    }
    cout<<"\n";
    for (int i = 0; i < 2 * edges; ++i) {
        cout<<E[i]<<" ";
    }
    cout<<"\n";


    float *bc = new float[nodes];
    // device pointers
    float *d_bc, *d_delta;
    int *d_V, *d_E, *d_d, *d_Q, *d_Q2, *d_S, *d_endpoints, *d_next_source;
    unsigned long long *d_sigma;

    int *next_source = new int;
    next_source[0] = 5;

    cudaMalloc((void**)&d_bc, sizeof(float) * nodes);
    cudaMalloc((void**)&d_V, sizeof(int) * (nodes + 1));
    cudaMalloc((void**)&d_E, sizeof(int) * (2*edges));
    cudaMalloc((void**)&d_Q, sizeof(int) * nodes * next_source[0]);
    cudaMalloc((void**)&d_Q2, sizeof(int) * nodes * next_source[0]);
    cudaMalloc((void**)&d_S, sizeof(int) * nodes * next_source[0]);
    cudaMalloc((void**)&d_endpoints, sizeof(int) * (nodes + 1) * next_source[0]);
    cudaMalloc((void**)&d_sigma, sizeof(unsigned long long) * nodes);
    cudaMalloc((void**)&d_d, sizeof(int) * nodes * next_source[0]);
    cudaMalloc((void**)&d_delta, sizeof(float) * nodes* next_source[0]);
    cudaMalloc((void**)&d_next_source, sizeof(int) * nodes);
    
    cudaMemcpy(d_V, V, sizeof(int) * (nodes+1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_E, E, sizeof(int) * (2*edges), cudaMemcpyHostToDevice);
	cudaMemset(d_bc, 0, sizeof(float) * nodes);
	cudaMemcpy(d_next_source , next_source, sizeof(int), cudaMemcpyHostToDevice);

    //Grid parameters
	dim3 dimBlock, dimGrid;
	dimGrid.x = 5;
	dimGrid.y = 1;
	dimGrid.z = 1;

	dimBlock.x = 64;
	dimBlock.y = 1;
	dimBlock.z = 1;
    betweenness_centrality_kernel <<<dimGrid, dimBlock>>> (d_bc, nodes, edges, d_V, d_E, d_Q, d_Q2, d_S, d_endpoints, d_sigma, d_d, d_delta, next_source);

    cudaMemcpy(bc, d_bc, sizeof(float) * nodes, cudaMemcpyDeviceToHost);
    
    cout<<"Res: \n";
    for (int i = 0; i < nodes; i++) {
        cout<<bc[i]<<" ";
    }
    cout<<endl;

    cudaFree(d_bc);
    cudaFree(d_V);
    cudaFree(d_E);
    cudaFree(d_S);
    cudaFree(d_Q);
    cudaFree(d_Q2);
    cudaFree(d_endpoints);
    cudaFree(d_sigma);
    cudaFree(d_d);
    cudaFree(d_delta);
    cudaFree(d_next_source);


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