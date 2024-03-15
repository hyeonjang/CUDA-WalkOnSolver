# CUDA-WalkOnSolver

### Project Overview
1. CUDA implementation of [Monte Carlo Geometry Processing](https://www.cs.cmu.edu/~kmcrane/Projects/MonteCarloGeometryProcessing/index.html)

### Included Features
1. GPU based Linear Bounding Volume Hierarchy, Closest point query
1. Laplace and Poisson PDE solver based on Walk on Sphere algorithm
2. Consideration of only Diriclet Boundary Condition
3. Parallel execution of 1's algorithm

### Future works 
- [ ] Other BVH method
- [ ] Other Walk on method (Walk on Star shape, Walk on Boundary)
- [ ] Shader excution reordering

### Build dependencies
CMake, C++17, CUDA 11.8, Visual Studio 2019 (Windows 10, 11)

### Third-party dependencies
[tinyply](https://github.com/ddiakopoulos/tinyply), [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)(reference), [Polyscope](https://polyscope.run/) (for visualiziation)

### Demo & Usage
- Visualization of the tested solver by [Polyscope](https://polyscope.run/) is implemented in demo/wosolver.cu. Please refer this.

Windows
```bash
mkdir build
cd build && cmake ..
cmake -B build
.\build\test_wos.exe
```
- Define Boundary Condition and Source Term for PDE
The functioncallity in this repo works by functor. Just define the interesting boundary condition and source term as below.
```cxx
struct BoundaryCondition {
    HOST_DEVICE float operator()(vec3 x) {
        return cos(2.f*M_PI*x.x) * sin(2.f*M_PI*x.y);
    }
};

struct SourceTerm {
    HOST_DEVICE float operator()(vec3 x) {
        return (M_PI*M_PI) * cos(2.f*M_PI*x.x) * sin(2.f*M_PI*x.y);
    }
};

int main() {
    WoSolver<TriangleLBVH> solver(mesh.v(), mesh.f(), stream);

    Laplace<BoundaryCondition> laplace;
    Poisson<BoundaryCondition, SourceTerm> poisson;
    
    ... make cuda stream interesting points
    ...

    auto gpu_result = solver.estimate_features(stream, gpu_points.mut_span(), laplace, poisson);
}

```


