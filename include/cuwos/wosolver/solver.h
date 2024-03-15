#pragma once

#include <pcg32.h>
#include <cuwos/feature.h>
#include <cuwos/wosolver/bvh.h>
#include <cuwos/wosolver/function.h>

namespace solver {
}
__constant__ int N_WALK = 64;
__constant__ int MAX_ITER = 128;

// buggy here
template <typename PDE>
using WalkFeature = Features<typename PDE::Value>;

template <typename ... PDE>
using WalkFeatures = Features<typename PDE::Value ...>;

// this only used for python-side code
template <class Tbvh>
class WoSolver {
public:

    WoSolver(){};
    WoSolver(gpu::span<vec3> v, gpu::span<ivec3> f, cudaStream_t stream)
    : m_stream(stream)
    , m_bvh(create_bvh<bvh_type::LBVH>(v, f, m_stream)) {};

    gpu::memory<vec3> sample_bounding_box_uniform(size_t size, cudaStream_t stream) const;
    gpu::memory<vec3> sample_object_inner_uniform(size_t intial_size, cudaStream_t stream) const;

    template <typename PDE>
    gpu::memory<typename PDE::Value> solve(cudaStream_t stream, gpu::span<vec3> pos, PDE& pde) const;

    // compute for multiple pde
    template <typename ... PDE>
    WalkFeatures<PDE...> estimate_features(cudaStream_t stream, gpu::span<vec3> pos, PDE&... pde) const;

private:
    template <typename PDE>
    void estimate(cudaStream_t stream, gpu::span<vec3> pos, PDE& pde, typename PDE::Value* result) const;

    // compute for multiple pde
    template <typename ... PDE>
    void estimate(cudaStream_t stream, gpu::span<vec3> pos, WalkFeatures<PDE...>& features, PDE&... pde) const;


private:
    cudaStream_t m_stream;
    std::unique_ptr<BVH<Tbvh>> m_bvh;
};

#if defined(__CUDACC__)
#include "cuwos/wosolver/solver.cuh"
#endif

// only need for python, for template instantiate
struct Sine {
    __host__ __device__ f32 operator()(vec3 x) {
        return cos(2.f*M_PI*x.x) * sin(2.f*M_PI*x.y);
        // return fmod(floor(8.0 * x.x) + floor(8.0 * x.y) + floor(8.0 * x.z), 2.0);
    }
};

// using LaplaceFeature = WalkFeatures<Laplace<Sine>>;