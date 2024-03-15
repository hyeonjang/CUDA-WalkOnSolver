#pragma once

#include <limits>

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

#include <cuwos/common.h>
#include <cuwos/memory.h>
#include <cuwos/wosolver/aabb.h>
#include <cuwos/wosolver/primitive.h>

using namespace tcnn;

enum class bvh_type {
    LBVH,
    NONE,
};

template <typename Derived, typename Prim> class BVH;
template <typename prim_t> class lbvh;

namespace QueryFunction {
    using Return = thrust::tuple<f32, bool, i32>; // distance, signed, index
    
    struct unsigned_distance_square {
        template <typename prim_t>
        HOST_DEVICE Return operator()(const prim_t& prim, const vec3 point) const {
            return {prim.distance_square(point), 0, -1};
        }
    };

    struct signed_distance_square {
        template <typename prim_t>
        HOST_DEVICE Return operator()(const prim_t& prim, const vec3 point) const {
            auto result = prim.signed_distance_square(point);
            return {thrust::get<0>(result), thrust::get<1>(result), -1};
        }
    };
}

struct QueryResult {

    QueryResult(size_t size)
    : distances(size), signs(size), indices(size) {}

    gpu::memory<f32> distances;
    gpu::memory<u8>  signs;
    gpu::memory<i32> indices;
};

std::vector<Triangle> permute_triangle_vertices_cpu(const cpu::span<vec3> v, const cpu::span<ivec3> f);
void permute_triangle_vertices_gpu(const gpu::span<vec3> v, const gpu::span<ivec3> f, Triangle* prims, cudaStream_t stream=nullptr);
gpu::memory<Triangle> permute_triangle_vertices_gpu(const gpu::span<vec3> v, const gpu::span<ivec3> f, cudaStream_t stream=nullptr);

template <typename T, int MAX_SIZE=32>
class FixedStack {
public:
	__host__ __device__ void push(T val) {
		if (m_count >= MAX_SIZE-1) {
			printf("WARNING TOO BIG\n");
		}
		m_elems[m_count++] = val;
	}

	__host__ __device__ T pop() {
		return m_elems[--m_count];
	}

	__host__ __device__ bool empty() const {
		return m_count <= 0;
	}

private:
	T m_elems[MAX_SIZE];
	int m_count = 0;
};

template <typename T, int SIZE=32>
class FixedQueue {
public:
    __host__ __device__ void push(T item) {
        if (full()) {
            printf("WARNING TOO BIG\n");
            return;
        }
        arr[rear] = item;
        rear = (rear + 1) % SIZE;
        count++;
    }

    __host__ __device__ T pop() {
        if (empty()) { 
            // cout << "Queue is empty" << endl;
            printf("EMPTY\n");
            return T();
        }

        T item = arr[front];
        front = (front + 1) % SIZE;
        count--;
        return item;
    }

	__host__ __device__ bool empty() const {
		return count == 0;
	}

    __host__ __device__ bool full() const {
        return count >= SIZE;
    }

private:
	T arr[SIZE];
    int front = 0, rear = 0, count = 0;
};

template <typename T, size_t SIZE=1024>
class PriorityQueue {
public:
    HOST_DEVICE PriorityQueue() : size(0) {}

    HOST_DEVICE void swap(T& a, T& b) {
        T temp = a;
        a = b;
        b = temp;
    }

    HOST_DEVICE void heapify(int i) {
        int smallest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (left < size && thrust::get<0>(heap[left]) < thrust::get<0>(heap[smallest]))
            smallest = left;

        if (right < size && thrust::get<0>(heap[right]) < thrust::get<0>(heap[smallest]))
            smallest = right;

        if (smallest != i) {
            swap(heap[i], heap[smallest]);
            heapify(smallest);
        }
    }

    HOST_DEVICE void push(T node) {
        if (size < SIZE) {
            int i = size;
            heap[size++] = node;

            while (i != 0 && thrust::get<0>(heap[(i - 1) / 2]) > thrust::get<0>(heap[i])) {
                swap(heap[i], heap[(i - 1) / 2]);
                i = (i - 1) / 2;
            }
        } else if (thrust::get<0>(node) < thrust::get<0>(heap[0])) {
            heap[0] = node;
            heapify(0);
        }
    }

    HOST_DEVICE T pop() {
        if (size <= 0)
            return thrust::make_tuple(std::numeric_limits<f32>::max(), -1);
        if (size == 1) {
            size--;
            return heap[0];
        }

        T root = heap[0];
        heap[0] = heap[--size];
        heapify(0);

        return root;
    }

    HOST_DEVICE bool empty() {
        return size == 0;
    }
private:
    T heap[SIZE];
    int size;
};

// CRTP approach
template <typename Derived, typename Prim = Triangle>
class BVH {
public:
    using aabb_t = typename Prim::AABB;

    BVH(){} 
    virtual ~BVH() {};

public:
    QueryResult closest_point_query(const gpu::span<vec3> points) {
        return static_cast<Derived*>(this)->closest_point_query_impl(points);
    }

    gpu::unique<Derived> device() {
        return static_cast<Derived*>(this)->device_impl();
    }

    // can be confusing
    DEVICE const auto* root_bb() const {
        return static_cast<const Derived*>(this)->root_bb_impl();
    }

};

// Linear bvh
namespace LBVH {
static constexpr u32 INVALID = 0xFFFFFFFF;
__constant__ static constexpr f32 MAX_DIST = 4000.0f;
} // namespace LBVH

template <typename prim_t>
class lbvh : public BVH<lbvh<prim_t>, prim_t> {
public:
    using Base = BVH<lbvh<prim_t>, prim_t>;
    using aabb_t = typename Base::aabb_t;

    lbvh(): Base(), prims(nullptr), nodes(nullptr), aabbs(nullptr){};
    lbvh(const gpu::span<vec3> v, const gpu::span<ivec3> f, cudaStream_t stream)
    :n_prims(f.size()), n_nodes(f.size()*2-1){ 

        // allocation
        allocation = std::make_shared<gpu::memory<u8>>(
                sizeof(prim_t) * n_prims + 
                sizeof(node_t) * n_nodes + 
                sizeof(aabb_t) * n_nodes
        );

        prims = reinterpret_cast<prim_t*>(allocation->data());
        nodes = reinterpret_cast<node_t*>(allocation->data() + sizeof(prim_t)*n_prims);
        aabbs = reinterpret_cast<aabb_t*>(allocation->data() + sizeof(prim_t)*n_prims + sizeof(node_t)*n_nodes);

        // permute and build
        permute_triangle_vertices_gpu(v, f, prims, stream);
        build_impl(stream); 

        // sync
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();
    };

    // ~lbvh() {
    //     allocation.reset();
    // }

    void build_impl(cudaStream_t stream);
    QueryResult closest_point_query_impl(const gpu::span<vec3> points);

public:
    struct node_t {

        HOST_DEVICE node_t(): parent(LBVH::INVALID), id(LBVH::INVALID), l(LBVH::INVALID), r(LBVH::INVALID) {};
        HOST_DEVICE node_t(u32 id): parent(LBVH::INVALID), id(id), l(LBVH::INVALID), r(LBVH::INVALID) {};

        u32 parent;
        u32 id;
        u32 l,r;

        constexpr HOST_DEVICE bool is_leaf() const {
            return id != LBVH::INVALID;
        };
    };

    DEVICE const aabb_t* root_bb_impl() const {
        return aabbs;
    }

    gpu::unique<lbvh> device_impl() {
        lbvh* device_ptr;
        CHECK_CUDA(cudaMalloc(&device_ptr, sizeof(lbvh)));
        CHECK_CUDA(cudaMemcpy(device_ptr, this, sizeof(lbvh), cudaMemcpyHostToDevice));
        return gpu::unique<lbvh>::create(device_ptr);
    }


// owner some memory permuted objects, e.g.) triangles
public:
    prim_t* prims;
    node_t* nodes;
    aabb_t* aabbs;
    size_t n_prims, n_nodes;

    // memory allocation
	std::shared_ptr<gpu::memory<uint8_t>> allocation;
};

using TriangleLBVH = lbvh<Triangle>;

// one options, V, F
inline lbvh<Triangle> create_lbvh(gpu::span<vec3> v, gpu::span<ivec3> f, cudaStream_t stream=nullptr) {
    auto bvh = lbvh<Triangle>(v, f, stream);
    return bvh;
}

// @@ factory pattern
template <bvh_type type>
auto create_bvh(gpu::span<vec3> v, gpu::span<ivec3> f, cudaStream_t stream=nullptr) {
    return std::make_unique<TriangleLBVH>(v, f, stream);
}

namespace LBVH {
template <typename prim_t, typename node_t, typename aabb_t, typename query_function>
DEVICE auto point_query(const vec3 point, const prim_t* prims, const node_t* nodes, const aabb_t* aabbs, const query_function& query) {

    using Return = QueryFunction::Return;

    PriorityQueue<thrust::tuple<f32, i32>> queue;
    queue.push({0.0f, 0});

    Return shortest = thrust::make_tuple(LBVH::MAX_DIST, 0, -1);

    f32 distance; i32 idx; 
    while (!queue.empty()) {
        thrust::tie(distance, idx) = queue.pop();

        if(distance > thrust::get<0>(shortest)) continue;

        const auto& node = nodes[idx];
        if (node.is_leaf()) {
            auto query_result = query(prims[node.id], point);  // Calculate distance to the primitive
            if (thrust::get<0>(query_result) < thrust::get<0>(shortest)) {
                shortest = query_result;
            }
        } else {
            f32 l = aabbs[node.l].longest_distance(point);
            f32 r = aabbs[node.r].longest_distance(point);
            
            queue.push({l, node.l});
            queue.push({r, node.r});
        }
    }
    return shortest;
}

template <typename prim_t>
DEVICE __forceinline__ f32 unsigned_distance(const lbvh<prim_t>* bvh, const vec3 pos) {
    auto result = point_query(pos, bvh->prims, bvh->nodes, bvh->aabbs, QueryFunction::unsigned_distance_square());
    return sqrt(thrust::get<0>(result));
}

template <typename prim_t>
DEVICE __forceinline__ thrust::tuple<f32, bool> signed_distance(const lbvh<prim_t>* bvh, const vec3 pos) {
    auto result = point_query(pos, bvh->prims, bvh->nodes, bvh->aabbs, QueryFunction::signed_distance_square());
    return {sqrt(thrust::get<0>(result)), thrust::get<1>(result)};
}
} // namespace LBVH

template <typename Derived>
DEVICE f32 unsigned_distance(const BVH<Derived>* bvh, vec3 pos) {
    return LBVH::unsigned_distance(static_cast<const Derived*>(bvh), pos);
}

template <typename Derived>
DEVICE thrust::tuple<f32, bool> signed_distance(const BVH<Derived>* bvh, vec3 pos) {
    return LBVH::signed_distance(static_cast<const Derived*>(bvh), pos);
}

// optix bvh
template <typename prim_t>
class obvh : public BVH<obvh<prim_t>, prim_t> {
public:
    using Base = BVH<obvh<prim_t>, prim_t>;

private:
    gpu::memory<prim_t> prims;
};