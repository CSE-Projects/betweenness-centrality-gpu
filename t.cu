#include <iostream>
#include <string>
#include <sstream>
#include <cuda.h>

#define ll long long 

using namespace std;

__global__
void vertex_parallel_sp(int nodes, int s, int *C, int *R, int *d, int *sigma) {
    int idx = threadIdx.x;
    //Initialize d and sigma
    for(int v=idx; v<nodes; v+=blockDim.x)
    {
      if(v == s)
      {
        d[v] = 0;
        sigma[v] = 1;
      }
      else
      {
        d[v] = INT_MAX;
        sigma[v] = 0;
      }
    }
    __shared__ int current_depth;
    __shared__ bool done;
    
    if(idx == 0)
    {
      done = false;
      current_depth = 0;
    }
    __syncthreads();
    
    //Calculate the number of shortest paths and the 
    // distance from s (the root) to each vertex
    while(!done)
    {
      __syncthreads();
      done = true;
      __syncthreads();
    
      for(int v=idx; v<nodes; v+=blockDim.x) //For each vertex...
      {
        if(d[v] == current_depth)
        {
          for(int r=R[v]; r<R[v+1]; r++) //For each neighbor of v
          {
            int w = C[r];
            if(d[w] == INT_MAX)
            {
              d[w] = d[v] + 1;
              done = false;
            }
            if(d[w] == (d[v] + 1))
            {
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
}

__global__
void vertex_parallel_sp(int nodes, int s, int *C, int *R, int *d, int *sigma) {
    int idx = threadIdx.x;
    //Initialize d and sigma
    for(int v=idx; v<nodes; v+=blockDim.x)
    {
      if(v == s)
      {
        d[v] = 0;
        sigma[v] = 1;
      }
      else
      {
        d[v] = INT_MAX;
        sigma[v] = 0;
      }
    }
    __shared__ int current_depth;
    __shared__ bool done;
    
    if(idx == 0)
    {
      done = false;
      current_depth = 0;
    }
    __syncthreads();
    
    //Calculate the number of shortest paths and the 
    // distance from s (the root) to each vertex
    while(!done)
    {
      __syncthreads();
      done = true;
      __syncthreads();
    
      for(int v=idx; v<nodes; v+=blockDim.x) //For each vertex...
      {
        if(d[v] == current_depth)
        {
          for(int r=R[v]; r<R[v+1]; r++) //For each neighbor of v
          {
            int w = C[r];
            if(d[w] == INT_MAX)
            {
              d[w] = d[v] + 1;
              done = false;
            }
            if(d[w] == (d[v] + 1))
            {
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
}


int main () {
    
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

    int *d_d, *d_sigma, *d_V, *d_E;

    cudaMalloc((void**)&d_d, sizeof(int) * nodes);
    cudaMalloc((void**)&d_sigma, sizeof(int) * nodes);
    cudaMalloc((void**)&d_V, sizeof(int) * (nodes + 1));
    cudaMalloc((void**)&d_E, sizeof(int) * (2*edges));

    cudaMemcpy(d_V, V, sizeof(int) * (nodes+1), cudaMemcpyHostToDevice);
	cudaMemcpy(d_E, E, sizeof(int) * (2*edges), cudaMemcpyHostToDevice);

    vertex_parallel_sp <<<1, 64>>> (nodes, 0, d_E, d_V, d_d, d_sigma);

    cudaMemcpy(d, d_d, sizeof(float) * nodes, cudaMemcpyDeviceToHost);
    cudaMemcpy(sigma, d_sigma, sizeof(float) * nodes, cudaMemcpyDeviceToHost);
    
    cout<<"Res: \n";
    for (int i = 0; i < nodes; i++) {
        cout<<d[i]<<" "<<sigma[i]<<endl;
    }
    cout<<endl;


    cudaFree(d_sigma);
    cudaFree(d_d);
    cudaFree(d_V);
    cudaFree(d_E);


    return 0;
}
