#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>

#include <cuwos/wosolver/bvh.h>

template <typename T, size_t N> __forceinline__
DEVICE void atomic_merge(BaseAABB<T, N>* lhs, const BaseAABB<T, N>& rhs) {
    atomic_min(&lhs->lower, rhs.lower);
    atomic_max(&lhs->upper, rhs.upper);
}

std::vector<Triangle> permute_triangle_vertices_cpu(const cpu::span<vec3> v, const cpu::span<ivec3> f) {
    std::vector<Triangle> triangles(f.size());
    for(size_t i=0; i<f.size(); i++) {
        triangles[i] = Triangle(v[f[i].x], v[f[i].y], v[f[i].z]);
    }
    return std::move(triangles);
}

namespace impl {
__global__ void permute_triangle(gpu::span<vec3> v, const gpu::span<ivec3> f, Triangle* triangles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= f.size()) return;

    triangles[i].v0 = v[f[i].x];
    triangles[i].v1 = v[f[i].y];
    triangles[i].v2 = v[f[i].z];
}
} // namespace impl

void permute_triangle_vertices_gpu(const gpu::span<vec3> v, const gpu::span<ivec3> f, Triangle* triangles, cudaStream_t stream) {
    impl::permute_triangle<<<n_blocks_linear(f.size()), N_THREADS_LINEAR, 0, stream>>>(
        v, f, triangles
    );
}

gpu::memory<Triangle> permute_triangle_vertices_gpu(const gpu::span<vec3> v, const gpu::span<ivec3> f, cudaStream_t stream) {
    gpu::memory<Triangle> triangles(f.size());
    impl::permute_triangle<<<n_blocks_linear(f.size()), N_THREADS_LINEAR, 0, stream>>>(
        v, f, triangles.data()
    );

    return std::move(triangles);
}

namespace LBVH {

template <typename U, size_t N> __forceinline__
HOST_DEVICE tvec<U, N> expand_bits(tvec<U, N> v) noexcept {
    TCNN_PRAGMA_UNROLL 
    for (int i=0; i<N; i++) {
        v[i] = (v[i] * 0x00010001u) & 0xFF0000FFu;
        v[i] = (v[i] * 0x00000101u) & 0x0F00F00Fu;
        v[i] = (v[i] * 0x00000011u) & 0xC30C30C3u;
        v[i] = (v[i] * 0x00000005u) & 0x49249249u;
    }
    return v;
}

template <typename T, size_t N> __forceinline__
HOST_DEVICE u64 morton_code(tvec<T, N> pos, f32 resolution = 1024.0f) noexcept {
    pos = min(max(pos * resolution, 0.0f), resolution - 1.0f);
    const auto bit = expand_bits<u64, N>(static_cast<tvec<u64, N>>(pos));
    u64 result = 0;
    TCNN_PRAGMA_UNROLL for(int i=N-1; i>=0; i--) {
        result += bit[i] * (1 << i);
    }
    return result;
}

//https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
inline __device__ std::pair<int, int> determine_range(const u64* mortons, const int i, const int num_prims) {
    
    if(i == 0) return std::make_pair(0, num_prims-1);

    const auto delta = [&mortons, &num_prims](const int self, const int next) {
        if(next < 0 || next >= num_prims) return -1;

        if(mortons[self] == mortons[next]) {
            return 32 + ::__clz(self ^ next);
        }
        return ::__clz(mortons[self] ^ mortons[next]);
    };

    // determine direction of the range (+1 or -1)
    const int delta_l = delta(i, i-1);
    const int delta_r = delta(i, i+1);

    // compute upper bound for the  length of the range
    int d; // direction
    int delta_min; // min of delta_r and delta_l
    if (delta_r < delta_l) {
        d = -1;
        delta_min = delta_r;
    } else {
        d = 1;
        delta_min = delta_l;
    }
    
    // compute upper bound of the length of the range
    unsigned int l_max = 2;
    while (delta(i, i + l_max * d) > delta_min) {
        l_max <<= 1;
    }

    // find other end using binary search
    unsigned int l = 0;
    for (unsigned int t = l_max >> 1; t > 0; t >>= 1) {
        if (delta(i, i + (l + t) * d) > delta_min) {
            l += t;
        }
    }
    int j = i + l * d;
    // printf("%d ", j);
    return std::minmax(i, j);
}

inline __device__ int find_split(const u64* mortons, const int first, const int last, const u32 num_prims) {

    const auto delta = [&mortons, &num_prims](const u32 self, const u32 next) -> int {
        if(next < 0 || next >= num_prims) return -1;

        if(mortons[self] == mortons[next]) {
            return 32 + ::__clz(self ^ next);
        }
        return ::__clz(mortons[self] ^ mortons[next]);
    };

    const int common_prefix = delta(first, last);

    // binary search
    int split = first;
    int step = last - first;

    do {
        step = (step + 1) >> 1;
        const int new_split = split + step;

        if(new_split < last) {
            const int split_prefix = delta(first, new_split);
            if(split_prefix > common_prefix) {
                split = new_split;
            }
        }
    } while(step > 1);
    return split;
}

template <typename prim_t, typename node_t, typename aabb_t>
__global__ void prepare_mortons(const u32 num_prims, const prim_t* prims, node_t* nodes, aabb_t* aabbs, aabb_t* root_bb, u64* mortons) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_prims) return;

    // better to fetch to __shared__
    nodes[idx] = node_t(idx);
    aabbs[idx] = prims[idx].aabb();
    __syncthreads();

    // merge whol bbox
    atomic_merge(root_bb, aabbs[idx]);
    __syncthreads();

    // mortons code
    auto center = aabbs[idx].centroid();

    using vec = decltype(center);

    center -= root_bb->lower;
    center /= (root_bb->upper - root_bb->lower);
    mortons[idx] = morton_code<vec::underlying_type, vec::N>(center);
}

template <typename node_t>
__global__ void build_nodes(const u32 num_prims, node_t* nodes, const u64* mortons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_prims - 1) return;

    nodes[idx] = node_t();

    const auto [l, r] = determine_range(mortons, idx, num_prims);
    const int split = find_split(mortons, l, r, num_prims);

    nodes[idx].l = split;
    nodes[idx].r = split + 1;
    if(l == split)   { nodes[idx].l += num_prims - 1; }
    if(r == split+1) { nodes[idx].r += num_prims - 1; }

    nodes[nodes[idx].l].parent = idx;
    nodes[nodes[idx].r].parent = idx;

    return;
}

template <typename node_t, typename aabb_t>
__global__ void build_aabbs(const u32 num_prims, int* flags, node_t* nodes, aabb_t* aabbs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = i + num_prims - 1;
    if(idx >= 2 * num_prims - 1) return;

    flags[i] = 0;
    __syncthreads();

    u32 parent = nodes[idx].parent;
    while(parent != LBVH::INVALID) {
        int visited = atomicCAS(&flags[parent], 0, 1);
        if (visited == 0) return;
        assert(visited == 1);

        aabbs[parent] = merge(aabbs[nodes[parent].l], aabbs[nodes[parent].r]);
        parent = nodes[parent].parent;
    }
} 
} // namespace LBVH

template <typename prim_t>
void lbvh<prim_t>::build_impl(cudaStream_t stream) {

    // 1. prepare
    const size_t num_prims = n_prims;
    const size_t num_internal_nodes = num_prims - 1;

    gpu::memory<u64> mortons(num_prims);
    gpu::memory<aabb_t> root_bb(1);
    LBVH::prepare_mortons<<<n_blocks_linear(num_prims), N_THREADS_LINEAR,  0, stream>>>(
        num_prims, prims, nodes + num_internal_nodes, aabbs + num_internal_nodes, root_bb.data(),
        mortons.data()
    );

    // one sorting
    thrust::sort_by_key(thrust::cuda::par.on(stream), 
        mortons.data(), mortons.data() + num_prims,
        thrust::make_zip_iterator(thrust::make_tuple(nodes + num_internal_nodes, aabbs + num_internal_nodes))    
    );

    // run build
    LBVH::build_nodes<<<n_blocks_linear(num_prims), N_THREADS_LINEAR, 0, stream>>>(
        num_prims, nodes, mortons.data()
    );

    gpu::memory<i32> flags(num_internal_nodes); flags.memset(0);
    LBVH::build_aabbs<<<n_blocks_linear(num_prims), N_THREADS_LINEAR, 0, stream>>>(
        num_prims, flags.data(), nodes, aabbs
    );
}

// here more flexbile functions needed
template <typename prim_t, typename node_t, typename aabb_t, typename... args_t>
__global__ void kernel_closest_point_query(const u32 counts, const vec3* query_points, const node_t* nodes, const aabb_t* aabbs, const prim_t* prims, 
    f32* distances, u8* signs, i32* indices) {
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= counts) return;

    thrust::tie(distances[idx], signs[idx], indices[idx]) 
        = LBVH::point_query<prim_t>(query_points[idx], prims, nodes, aabbs, QueryFunction::unsigned_distance_square());
}

template <typename prim_t>
QueryResult lbvh<prim_t>::closest_point_query_impl(const gpu::span<vec3> points) {

    QueryResult results(points.size());
    kernel_closest_point_query<<<n_blocks_linear(points.size()), N_THREADS_LINEAR>>>(
        points.size(), points.data(), nodes, aabbs, prims,
        results.distances.data(), results.signs.data(), results.indices.data()
    );
    return results;
}

template void lbvh<Triangle>::build_impl(cudaStream_t stream);
template QueryResult lbvh<Triangle>::closest_point_query_impl(const gpu::span<vec3> points);