# Betweenness Centrality on GPU
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

3. Parallel Implementation using using Work-efficient Method
    ```
    $ nvcc p_imp_1.cu
    $ ./a.out < graph10p4
    ```

4. Parallel Implementation using Vertex-parallel Method
    ```
    $ nvcc p_imp_2.cu
    $ ./a.out < graph10p4
    ```

## File Structure
- `p_imp_1.cu` - Parallel Implementation using Work-efficient Method
- `p_imp_2.cu` - Parallel Implementation using Vertex-parallel Method
- `serial.cc` - Serial implementation
- `input_format/` - Contains sample input format
- `results/` - Results for test from graph inputs sizes mentioned in Report

## Results and Summary
Mentioned in the Report

## References
- https://devblogs.nvidia.com/accelerating-graph-betweenness-centrality-cuda/
- http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.728.2926&rep=rep1&type=pdf
