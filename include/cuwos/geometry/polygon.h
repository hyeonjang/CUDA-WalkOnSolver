#pragma once

#include <cuwos/common.h>
#include <cuwos/vector.h>
#include <cuwos/memory.h>

using namespace tcnn;

template <typename T> using VertexArray = gpu::memory<T>;
template <typename T> using FaceArray = gpu::memory<T>;
template <typename T> using EdgeArray = gpu::memory<T>;
template <typename T> using HalfEdgeArray = gpu::memory<T>;
template <typename T> using CornerArray = gpu::memory<T>;

// contain mesh structure, no algorithm
template <typename T, typename U, uint8_t D>
class BaseMesh {
public:
    using Tvec = tvec3<T>;
    using Uvec = tvec<U, D>;

    static const uint8_t static_vertex_dimension = 3;
    static const uint8_t static_face_dimension = D;

    BaseMesh(size_t nv, size_t nf): m_vertices(VertexArray<Tvec>(nv)), m_faces(FaceArray<Uvec>(nf)) {}

    void copy_from_host(const T* vdata, const U* udata) {
        m_vertices.copy_from_host((Tvec*)vdata, nV());
        m_faces.copy_from_host((Uvec*)udata, nF());
    }

    auto& vert() { return m_vertices; }
    auto& face() { return m_faces; }

    // more think here
    gpu::span<Tvec> v() const { return gpu::span<Tvec>{m_vertices.data(), m_vertices.size()}; }
    gpu::span<Uvec> f() const { return gpu::span<Uvec>{m_faces.data(), m_faces.size()}; }

    uint32_t nV() const { return m_vertices.size(); }
    uint32_t nF() const { return m_faces.size(); }
    uint32_t nE() const { return /*check manifold*/ nV() + nF() - 1; }

private:
    VertexArray<Tvec>   m_vertices;
    FaceArray<Uvec>     m_faces;
};

template <typename T, typename U>
class TriangleMesh : public BaseMesh<T, U, 3> {
public:
    using Base = BaseMesh<T, U, 3>;

    TriangleMesh(size_t nv, size_t nf): BaseMesh<T, U, 3>(nv, nf) {}
    static TriangleMesh<T, U> from_cpu_data(const std::vector<T>& vertices, const std::vector<U>& faces) {
        const size_t nv = vertices.size()/Base::static_vertex_dimension;
        const size_t nf = faces.size()/Base::static_face_dimension;
        // const size_t ne = nv + nf - 1; // by euler formular

        TriangleMesh mesh(nv, nf);
        mesh.copy_from_host(&vertices[0], &faces[0]);
        return mesh;
    }
};

template <typename T, typename U>
void sample_uniform(const TriangleMesh<T, U>& mesh, VertexArray<tvec3<T>>& points);

template <typename T, typename U>
void sample_poisson_disk(const TriangleMesh<T, U>& mesh, VertexArray<tvec3<T>>& points);


// read write meshes
namespace io {

#include <vector>
#include <string>

template <typename T, typename U>
std::pair<std::vector<T>, std::vector<U>> read_ply(const std::string& filepath);

template <typename T, typename U>
void write_triangle_mesh_ply(const std::string& filepath, cpu::span<tvec3<T>> v, cpu::span<tvec3<U>> f);

template <typename T, typename U>
void write_point_cloud_ply(const std::string& filepath, cpu::span<tvec3<T>> points, cpu::span<tvec3<U>> colors=nullptr);
}