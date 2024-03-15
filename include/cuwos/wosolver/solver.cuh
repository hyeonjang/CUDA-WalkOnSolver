#ifndef SOLVER_CUH
#define SOLVER_CUH

#include <thrust/remove.h>
#include <cuwos/wosolver/function.h>

// prepare
template <typename Tbvh>
__global__ void launch_sample_bounding_box_uniform(const size_t size, const Tbvh* bvh, pcg32* sampler, vec3* positions) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size) return;

    __shared__ const typename Tbvh::aabb_t* root_bb;
    __shared__ vec3 scale, lower;
    if (threadIdx.x == 0) {
        root_bb = bvh->root_bb();
        lower = root_bb->lower;
        scale = root_bb->upper - root_bb->lower;
    }
    __syncthreads();

    sampler->advance(idx * 3);
    vec3 sample_pos(sampler->next_float(), sampler->next_float(), sampler->next_float());
    positions[idx] = sample_pos * scale + lower;
}

template <typename Tbvh>
__global__ void launch_sample_object_inner_uniform(const size_t size, const Tbvh* bvh, pcg32* sampler, vec3* positions) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size) return;

    __shared__ const typename Tbvh::aabb_t* root_bb;
    __shared__ vec3 scale, lower;
    if (threadIdx.x == 0) {
        root_bb = bvh->root_bb();
        lower = root_bb->lower;
        scale = root_bb->upper - root_bb->lower;
    }
    __syncthreads();

    sampler->advance(idx * 3);
    vec3 sample_pos(sampler->next_float(), sampler->next_float(), sampler->next_float());
    sample_pos = sample_pos * scale + lower;
    
    f32 _; bool is_inner = false;
    thrust::tie(_, is_inner) = signed_distance(bvh, sample_pos);
    if(is_inner) {
        positions[idx] = sample_pos;
    } else {
        positions[idx] = vec3(0.0);
    }
}

template <typename Tbvh>
gpu::memory<vec3> WoSolver<Tbvh>::sample_bounding_box_uniform(size_t size, cudaStream_t stream) const {

    auto sampler = gpu::make_unique<pcg32>();

    gpu::memory<vec3> positions(size);
    launch_sample_bounding_box_uniform<Tbvh><<<n_blocks_linear(size), N_THREADS_LINEAR, 0, stream>>>(
        size, m_bvh->device().data(), sampler.data(), positions.data());
    return std::move(positions);
}

template <typename Tbvh>
gpu::memory<vec3> WoSolver<Tbvh>::sample_object_inner_uniform(size_t size, cudaStream_t stream) const {

    auto sampler = gpu::make_unique<pcg32>();

    // sample
    gpu::memory<vec3> positions(size);
    launch_sample_object_inner_uniform<Tbvh><<<n_blocks_linear(size), N_THREADS_LINEAR, 0, stream>>>(
        size, m_bvh->device().data(), sampler.data(), positions.data());

    // remove outer
    vec3* last = thrust::remove(thrust::cuda::par.on(stream), positions.data(), positions.data() + size, vec3(0.0f));
    positions.change_size_cpu(last - positions.data());

    //@@ is this proper?
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    return std::move(positions);
}

//
// wosolver
// considerable time complexity: log(epsilon) * log(V) + log(V)
template <typename Tbvh, typename PDE>
__device__ typename PDE::Value walk_on_sphere(const Tbvh* bvh, const vec3 init, pcg32* rng, PDE& pde) {
    
    typename PDE::Value sum(0.0f);

    vec3 x = init;
    f32 R = 100.f; bool is_inner = false;
    for(int i=0; i<MAX_ITER; i++) {
        thrust::tie(R, is_inner) = signed_distance(bvh, x);
        if(R < 1e-6) break;

        Sphere s(x, R);
        x = s.sample_surface(rng);
    }
    // rich to boundary
    return sum + pde.g(x);
}

// template <typename Tbvh, typename PDE>
// __device__ WalkFeature<PDE>::Type wos(const Tbvh* bvh, const vec3 init, pcg32* rng, PDE pde) {
    
//     typename PDE::Value sum(0.0);
//     vec3 gradient(0.0);

//     vec3 x = init;

//     // initial sample
//     Sphere s(x, unsigned_distance(bvh, x));
//     auto [y, r] = s.sample_volume(rng);
//     vec3 v = (y-x)/s.r;

//     f32 R = s.r; bool is_inner = false;
//     for(int i=0; i<MAX_ITER; i++) {
//         if(R < 1e-6) break;

//         // if constexpr(std::is_same_v<PDE, Laplace<typename PDE::BoundaryCondition>> == false) {
//         //     sum += pde(s.p, s.r, y, r);
//         //     gradient += pde.gradient(s.p, s.r, y, r);
//         //     thrust::tie(y, r) = s.sample_volume(rng);
//         // }

//         x = s.sample_surface(rng);
//         thrust::tie(R, is_inner) = signed_distance(bvh, x);
//         if(!is_inner) break;

//         s = Sphere(x, R);
//     }

//     // auto boundary_value = pde.g(x);
//     return WalkFeature<PDE>::init(pde.g(x));
//     // return WalkFeatures<PDE>::make(sum + boundary_value, gradient + boundary_value*v);
// }

template <typename Tbvh, typename ... PDE>
__device__ typename WalkFeatures<PDE...>::Type wos(const Tbvh* bvh, vec3 x, pcg32* rng, PDE... pde) {
    
    auto sum = WalkFeatures<PDE...>::zero();

    // initial sample
    Sphere s(x, unsigned_distance(bvh, x));
    auto [y, r] = s.sample_volume(rng);
    vec3 v = (y-x)/s.r;

    f32 R = s.r; bool is_inner = false;
    for(int i=0; i<MAX_ITER; i++) {
        if(R < 1e-6) break;

        // if constexpr(std::is_same_v<PDE, Laplace<typename PDE::BoundaryCondition>> == false) {
            sum = sum + thrust::make_tuple(pde(s.p, s.r, y, r)...);
            thrust::tie(y, r) = s.sample_volume(rng);
        // }

        x = s.sample_surface(rng);
        thrust::tie(R, is_inner) = signed_distance(bvh, x);
        if(!is_inner) break;

        s = Sphere(x, R);
    }
    return sum + thrust::make_tuple(pde.g(x)...);
}

template <typename Tbvh, typename PDE>
__global__ void launch_solver(const size_t size, const vec3* positions, pcg32* _rng, Tbvh* _bvh, PDE& _pde, typename PDE::Value* results) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size) return;

    // fetch
    __shared__ const Tbvh* bvh;
    __shared__ pcg32* rng;
    // __shared__ PDE pde;
    if(threadIdx.x == 0) {
        bvh = _bvh;
        rng = _rng;
        // pde = *_pde;
    }
    __syncthreads();

    printf("%d ", idx);
    // run
    typename PDE::Value sum(0.0f);
    rng->advance(idx * N_WALK);
    for(int i=0; i<N_WALK; i++) {
        sum += walk_on_sphere<Tbvh, PDE>(bvh, positions[idx], rng, _pde);
    }
    results[idx] = sum/PDE::Value(N_WALK);
}

template <typename Tbvh, typename ...PDE>
__global__ void launch_solver(const size_t size, const vec3* positions, pcg32* _rng, Tbvh* _bvh, typename WalkFeatures<PDE...>::Iter results, PDE&... pde) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size) return;

    // fetch
    __shared__ const Tbvh* bvh;
    __shared__ pcg32* rng;
    // __shared__ PDE pde;
    if(threadIdx.x == 0) {
        bvh = _bvh;
        rng = _rng;
        // pde = _pde;
    }
    __syncthreads();

    // run
    auto sum = WalkFeatures<PDE...>::zero();
    rng->advance(idx * N_WALK);
    for(int i=0; i<N_WALK; i++) {
        sum = sum + wos<Tbvh, PDE...>(bvh, positions[idx], rng, pde...);
    }
    results[idx] = sum/(f32)N_WALK;
}

template <typename Tbvh>
template <typename PDE>
gpu::memory<typename PDE::Value> WoSolver<Tbvh>::solve(cudaStream_t stream, gpu::span<vec3> positions, PDE& pde) const {
    
    gpu::memory<typename PDE::Value> result(positions.size());

    // run solver
    estimate(stream, positions, pde, result.data());

    return result;
}

template <typename Tbvh>
template <typename ... PDE>
WalkFeatures<PDE...> WoSolver<Tbvh>::estimate_features(cudaStream_t stream, gpu::span<vec3> positions, PDE&... pde) const {

    WalkFeatures<PDE...> features(positions.size());

    // run solver
    estimate(stream, positions, features, pde...);
    return features;
}

template <typename Tbvh>
template <typename PDE>
void WoSolver<Tbvh>::estimate(cudaStream_t stream, gpu::span<vec3> positions, PDE& pde, typename PDE::Value* result) const {
    
    size_t size = positions.size();

    // run solver
    auto sampler = gpu::make_unique<pcg32>();
    launch_solver<Tbvh, PDE><<<n_blocks_linear(size), N_THREADS_LINEAR, 0, stream>>>(
        size, positions.data(), sampler.data(), m_bvh->device().data(), pde, result
    );
}

template <typename Tbvh>
template <typename ... PDE>
void WoSolver<Tbvh>::estimate(cudaStream_t stream, gpu::span<vec3> positions, WalkFeatures<PDE...>& features, PDE& ... pde) const {
    
    size_t size = positions.size();

    // run solver
    auto sampler = gpu::make_unique<pcg32>();
    launch_solver<Tbvh, PDE...><<<n_blocks_linear(size), N_THREADS_LINEAR, 0, stream>>>(
        size, positions.data(), sampler.data(), m_bvh->device().data(), features.iter(), pde...
    );

    cudaStreamSynchronize(stream);
}


#endif // SOLVER_CUH