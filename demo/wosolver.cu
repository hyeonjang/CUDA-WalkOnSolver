#include <filesystem>

#include <cuwos/geometry/polygon.h>
#include <cuwos/wosolver/solver.h>

#include "polyscope/polyscope.h"
#include "polyscope/point_cloud.h"

// file reads
namespace fs = std::filesystem;
static auto current_folder = fs::path(__FILE__).parent_path();
static auto spot = (current_folder/"assets/spot.ply").string();

template <typename T>
std::vector<T> to_cpu_vector(gpu::memory<T>& mem) {
    std::vector<T> cpu_vector(mem.size());
    mem.copy_to_host(cpu_vector);
    return cpu_vector;
}

template <typename T>
std::vector<T> to_cpu_vector(const T* gpu_ptr, const size_t size) {
    CHECK(gpu_ptr);

    std::vector<T> cpu_vector(size);
    CHECK_CUDA(cudaMemcpy(cpu_vector.data(), gpu_ptr, sizeof(T)*size, cudaMemcpyDeviceToHost));
    return cpu_vector;
}

template <typename T>
void add_quantity(std::string name, const std::vector<T>& data, size_t index) {
    if constexpr (std::is_same_v<T, f32>) {
        polyscope::getPointCloud(name)->addScalarQuantity(fmt::format("index {}", index), data);
    } else {
        polyscope::getPointCloud(name)->addVectorQuantity(fmt::format("index {}", index), data);
    }
}

template <typename ... Args, size_t ... Is>
void register_features_impl(std::string name, Features<Args...>& features, std::index_sequence<Is...>) {
    (add_quantity(name, to_cpu_vector(thrust::get<Is>(features.data()), features.size()), Is), ...);
}

template <typename ... Args>
void register_features(std::string name, Features<Args...>& features) {
    register_features_impl(name, features, std::index_sequence_for<Args...>());
}

template <typename T>
void register_features(std::string name, gpu::memory<T>& features) {
    add_quantity(name, to_cpu_vector(features), 0);
}

struct Line {
    __host__ __device__ f32 operator()(vec3 x) {
        return cos(2.f*M_PI*x.x) * sin(2.f*M_PI*x.y);
        // return fmod(floor(8.0 * x.x) + floor(8.0 * x.y) + floor(8.0 * x.z), 2.0);
    }
};

struct Pose {
    __host__ __device__ f32 operator()(vec3 x) {
        // return 1.0;
        return (M_PI*M_PI) * cos(2.f*M_PI*x.x) * sin(2.f*M_PI*x.y);
    }
};


int main() {
    auto [v, f] = io::read_ply<f32, i32>(spot);
    auto mesh = TriangleMesh<f32, i32>::from_cpu_data(v, f);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    WoSolver<TriangleLBVH> solver(mesh.v(), mesh.f(), stream);

    Laplace<Line> laplace;
    Poisson<Line, Pose> poisson;

    auto gpu_points = solver.sample_object_inner_uniform(100000, stream);
    auto cpu_points = to_cpu_vector(gpu_points);

    auto gpu_result = solver.estimate_features(stream, gpu_points.mut_span(), laplace, poisson);

    polyscope::init();
    polyscope::registerPointCloud("my points", cpu_points);
    register_features("my points", gpu_result);
    polyscope::show();
}