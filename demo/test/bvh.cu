#include <gtest/gtest.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cuwos/geometry/polygon.h>
#include <cuwos/wosolver/bvh.h>
#include <cuwos/wosolver/solver.h>
#include <cuwos/wosolver/function.h>

#include <vector>
#include <filesystem>

// there is no code that directly compares the floating vectors in gtest
template <typename T>
void ExpectVectorsNear(const std::vector<T>& v1, const std::vector<T>& v2, T tolerance) {
    ASSERT_EQ(v1.size(), v2.size()) << "Vectors are of unequal length";

    for (size_t i = 0; i < v1.size(); ++i) {
        EXPECT_NEAR(v1[i], v2[i], tolerance) << "Vectors differ at index " << i;
    }
}

// file reads
namespace fs = std::filesystem;
static auto current_folder = fs::path(__FILE__).parent_path().parent_path();
static auto tetrahedron = (current_folder/"assets/tetrahedron.ply").string();
static auto bunny_decimate = (current_folder/"assets/bunny_decimate.ply").string();
static auto bunny = (current_folder/"assets/bunny.ply").string();
static auto cube = (current_folder/"assets/cube.ply").string();
static auto sofa = (current_folder/"assets/sofa.ply").string();
static auto spot = (current_folder/"assets/spot.ply").string();

// test fixed size stack and queue
TEST(FixedQueue, Test) {
    FixedStack<i32, 4> stack;
    FixedQueue<i32, 4> queue;
    for(int i=0; i<4; i++) {
        stack.push(i);
        queue.push(i);
    }

    std::vector<i32> r_stack, r_queue;
    for(int i=0; i<4; i++) {
        r_stack.push_back(stack.pop());
        r_queue.push_back(queue.pop());
    }

    ExpectVectorsNear(r_stack, std::vector<i32>{3, 2, 1, 0}, 0);
    ExpectVectorsNear(r_queue, std::vector<i32>{0, 1, 2, 3}, 0);
}

TEST(PriorityQueue, Test) {
    PriorityQueue<thrust::tuple<f32, i32>> queue;

    size_t size = 10;
    for(int i=0; i<size; i++) {
        queue.push(thrust::make_tuple((f32)(size-i), i));
    } 

    std::vector<i32> r_queue;
    std::vector<f32> rf_queue;
    for(int i=0; i<4; i++) {
        auto pop = queue.pop();
        rf_queue.push_back(thrust::get<0>(pop));
        r_queue.push_back(thrust::get<1>(pop));
    }

    auto ianswer = std::vector<i32>{9, 8, 7, 6};
    EXPECT_EQ(r_queue, ianswer);

    auto fanswer = std::vector<f32>{1.0f, 2.0f, 3.0f, 4.0f};
    ExpectVectorsNear(rf_queue, fanswer, 0.0f);
} 

// 
TEST(Triangle, Distance) {

    Triangle triangle;
    triangle.v0 = vec3{-21.0, -2.0, 0.0};
    triangle.v1 = vec3{1.0, 6.0, 1.0}; 
    triangle.v2 = vec3{0.0, 2.0, 0.0};

    vec3 p0{0.0, 0.0, 0.0};
    EXPECT_EQ(triangle.distance_square(p0), length2(triangle.closest_point(p0) - p0));

    vec3 p1{1.2, 0.1, 3.0};
    EXPECT_EQ(triangle.distance_square(p1), triangle.closest_point_distance(p1).second);

    vec3 p2{10.0, -10.0, 1.0};
    EXPECT_EQ(triangle.distance_square(p2), triangle.closest_point_distance(p2).second);
}

static pcg32 rng;
static std::vector<vec3> cpu_points(100);

class SolverTest : public testing::Test {

protected:

    SolverTest()
    : vf(io::read_ply<f32, i32>(spot))
    , mesh(TriangleMesh<f32, i32>::from_cpu_data(vf.first, vf.second))
    , gpu_points(cpu_points.size()) {
        CHECK_CUDA(cudaStreamCreate(&stream));
        // test 1
        bvh = create_lbvh(mesh.v(), mesh.f(), stream);
        
        // give initial points
        rng.advance(10*3);
        for(auto& p:cpu_points) {
            p = vec3{
                rng.next_float(),
                rng.next_float(),
                rng.next_float() 
            } * 2.0f - vec3(1.0f);
        }

        gpu_points.resize(cpu_points.size());
        gpu_points.copy_from_host(cpu_points);
    }

    template <typename T>
    std::vector<T> to_cpu_vector(gpu::memory<T>& mem) {
        std::vector<T> cpu_vector(mem.size());
        mem.copy_to_host(cpu_vector);
        return cpu_vector;
    }

    // query
    gpu::memory<vec3> gpu_points;

    // struct 
    cudaStream_t stream;
    std::pair<std::vector<f32>, std::vector<i32>> vf;
    TriangleMesh<f32, i32> mesh;
    TriangleLBVH bvh;
};

// build and run test
TEST_F(SolverTest, LBVH) {

    // mesh 
    auto cpu_search_non = [&, &v=vf.first, &f=vf.second]() {

        std::vector<Triangle> prims = permute_triangle_vertices_cpu(
            cpu::span<vec3>{(vec3*)v.data(), v.size()/3}, 
            cpu::span<ivec3>{(ivec3*)f.data(), f.size()/3}
        );

        // real search
        std::vector<f32> shortest;
        for(size_t j=0; j<cpu_points.size(); j++) {

            float distance = LBVH::MAX_DIST;
            for(size_t i=0; i<prims.size(); i++) {
                auto result = prims[i].distance_square(cpu_points[j]);
                if(result < distance) {
                    distance = result;
                }
            }
            shortest.push_back(distance);
        }
        return shortest;
    };

    // lbvh closest point query
    auto gpu_search_bvh = [&]() {
        auto gpu_result = bvh.closest_point_query(gpu::span<vec3>{gpu_points.data(), gpu_points.size()});
        return to_cpu_vector(gpu_result.distances);
    };

    // check results
    // EXPECT_EQ(cpu_search_non(), gpu_search_bvh());
    ExpectVectorsNear(cpu_search_non(), gpu_search_bvh(), 1e-6f);
}