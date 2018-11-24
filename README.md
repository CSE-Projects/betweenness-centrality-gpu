# Betweenness Centrality GPU
Betweenness Centrality for large sparse graphs on GPU using CUDA

## Team:
- Dibyadarshan Hota 16CO154
- Omkar Prabhu 16CO233

## Usage
1. Random Graph Generator
    ```
    $ g++ g_generator.cpp
    $ ./a.out > graph10p4
    65536 65536
    ```

2. Serial Implementation
    ```
    $ g++ serial.cc
    $ ./a.out < graph10p4
    ```

3. Parallel Implementation using using Work-efficient Method(p_imp_1)
    ```
    $ nvcc main_work_efficient_parallel.cu
    $ ./a.out < graph10p4
    ```

4. Parallel Implementation using Vertex-parallel Method (p_imp_2)
    ```
    $ nvcc main_vertex_parallel.cu
    $ ./a.out < graph10p4
    ```

## File Structure
#### Code:
- `main_work_efficient_parallel.cu` or `p_imp_1.cu` - Parallel Implementation using Work-efficient Method
- `main_vertex_parallel.cu` or `p_imp_2.cu` - Parallel Implementation using Vertex-parallel Method
- `main_vertex_parallel-serial.cu`
- `serial.cc` - Serial implementation
- `g_generator.cpp` - Random Gaph Generator Our Implementation
- `parse.py` - Convert 1 to 0 based node index
- `final_generator.cpp` - Random Gaph Generator used by the class
- `parse.c++` - Parse output from final_generator to covert to our input format

#### Results and Inputs:
- `results-common-graphs/` - Results for test for common graphs for the class
- `results-g-generator/` - Results for test for graphs from our graph generator
- `input_format/` - Contains sample input format

## Results and Summary
Mentioned in the Report

## References
- https://devblogs.nvidia.com/accelerating-graph-betweenness-centrality-cuda/
- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.728.2926&rep=rep1&type=pdf
