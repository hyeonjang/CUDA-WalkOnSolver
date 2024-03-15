#include <cuwos/geometry/polygon.h>

#include <fmt/format.h>

#define TINYPLY_IMPLEMENTATION
#include <tinyply.h>

#include <vector>
#include <thread>
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <exception>

namespace io {

using namespace tinyply;

// all these from tinyply example
struct memory_buffer : public std::streambuf {
    char * p_start {nullptr};
    char * p_end {nullptr};
    size_t size;

    memory_buffer(char const * first_elem, size_t size)
        : p_start(const_cast<char*>(first_elem)), p_end(p_start + size), size(size) {
        setg(p_start, p_start, p_end);
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) override {
        if (dir == std::ios_base::cur) gbump(static_cast<int>(off));
        else setg(p_start, (dir == std::ios_base::beg ? p_start : p_end) + off, p_end);
        return gptr() - p_start;
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
        return seekoff(pos, std::ios_base::beg, which);
    }
};

struct memory_stream : virtual memory_buffer, public std::istream {
    memory_stream(char const * first_elem, size_t size)
        : memory_buffer(first_elem, size), std::istream(static_cast<std::streambuf*>(this)) {}
};

// same manner as examples in tinyply
static inline std::vector<u8> read_file_binary(const std::string& filepath) {

    std::ifstream file(filepath, std::ios::binary);
    std::vector<u8> byte_buffer;
    CHECK(file.is_open(), "could not open binary ifstream to " + filepath);

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    byte_buffer.resize(size);
    
    CHECK(file.read((char*)byte_buffer.data(), size));
    return byte_buffer;
}

//@@ more complicated logic for others (point cloud, tetrahedron, quads ...)
template <typename T, typename U>
std::pair<std::vector<T>, std::vector<U>> read_ply(const std::string& filepath) {
    
    std::unique_ptr<std::istream> file_stream;
    std::vector<uint8_t> byte_buffer;

    try {
        byte_buffer = read_file_binary(filepath);
        file_stream.reset(new memory_stream((char*)byte_buffer.data(), byte_buffer.size()));
        // file_stream.reset(new std::ifstream(filepath, std::ios::binary));

        if (!file_stream || file_stream->fail()) throw std::runtime_error("file_stream failed to open " + filepath);

        file_stream->seekg(0, std::ios::end);
        const float size_mb = file_stream->tellg() * float(1e-6);
        file_stream->seekg(0, std::ios::beg);

        PlyFile file;
        file.parse_header(*file_stream);

        // Because most people have their own mesh types, tinyply treats parsed data as structured/typed byte buffers. 
        // See examples below on how to marry your own application-specific data structures with this one. 
        std::shared_ptr<PlyData> vertices, normals, colors, texcoords, faces, tripstrip;

        // The header information can be used to programmatically extract properties on elements
        // known to exist in the header prior to reading the data. For brevity of this sample, properties 
        // like vertex position are hard-coded: 
        try { vertices = file.request_properties_from_element("vertex", { "x", "y", "z" }); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        // try { normals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); }
        // catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        // try { colors = file.request_properties_from_element("vertex", { "red", "green", "blue", "alpha" }); }
        // catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        // try { colors = file.request_properties_from_element("vertex", { "r", "g", "b", "a" }); }
        // catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        // try { texcoords = file.request_properties_from_element("vertex", { "u", "v" }); }
        // catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        try { faces = file.request_properties_from_element("face", { "vertex_indices" }); }
        catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        // try { tripstrip = file.request_properties_from_element("tristrips", { "vertex_indices" }, 0); }
        // catch (const std::exception& e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

        file.read(*file_stream);

        std::vector<T> buff_vert(vertices->count * 3);
        std::memcpy((T*)buff_vert.data(), vertices->buffer.get(), vertices->buffer.size_bytes());

        std::vector<U> buff_face(faces->count * 3);
        std::memcpy((U*)buff_face.data(), faces->buffer.get(), faces->buffer.size_bytes());

        return std::make_pair(buff_vert, buff_face);
    } catch (const std::exception & e) {
        std::cerr << e.what() << std::endl;
    }
}

template <typename T, typename U>
void write_triangle_mesh_ply(const std::string& filename, cpu::span<tvec3<T>> v, cpu::span<tvec3<U>> f) {
    
    std::filebuf fb_binary;
    fb_binary.open(filename + "-binary.ply", std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) throw std::runtime_error("failed to open " + filename);

    std::filebuf fb_ascii;
    fb_ascii.open(filename + "-ascii.ply", std::ios::out);
    std::ostream outstream_ascii(&fb_ascii);
    if (outstream_ascii.fail()) throw std::runtime_error("failed to open " + filename);

    PlyFile file;
    file.add_properties_to_element("vertex", { "x", "y", "z" }, 
        Type::FLOAT32, v.size(), reinterpret_cast<uint8_t*>(v.data()), Type::INVALID, 0);

    file.add_properties_to_element("face", { "vertex_indices "}, 
        Type::UINT32, f.size(), reinterpret_cast<u8*>(f.data()), Type::UINT32, 3);

    file.get_comments().push_back("generated by tinyply 2.3");
    file.write(outstream_ascii, false);
    file.write(outstream_binary, true);
}

template <typename T, typename U>
void write_point_cloud_ply(const std::string& filename, const cpu::span<tvec3<T>> points, const cpu::span<tvec3<U>> colors) {

    std::filebuf fb_binary;
    fb_binary.open(filename + "-binary.ply", std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) throw std::runtime_error("failed to open " + filename);

    std::filebuf fb_ascii;
    fb_ascii.open(filename + "-ascii.ply", std::ios::out);
    std::ostream outstream_ascii(&fb_ascii);
    if (outstream_ascii.fail()) throw std::runtime_error("failed to open " + filename);

    PlyFile file;
    file.add_properties_to_element("vertex", { "x", "y", "z" }, 
        Type::FLOAT32, points.size(), reinterpret_cast<uint8_t*>(points.data()), Type::INVALID, 0);

    if(colors.data() != nullptr) {
        file.add_properties_to_element("vertex", { "red", "green" , "blue" }, 
                Type::UINT32, colors.size(), reinterpret_cast<uint8_t*>(colors.data()), Type::INVALID, 0);
    }

    file.get_comments().push_back("generated by tinyply 2.3");
    file.write(outstream_ascii, false);
    file.write(outstream_binary, true);
}

template std::pair<std::vector<f32>, std::vector<i32>> read_ply(const std::string& filepath); 
template std::pair<std::vector<f32>, std::vector<u32>> read_ply(const std::string& filepath);

template void write_point_cloud_ply(const std::string& filepath, const cpu::span<tvec3<f32>> points, const cpu::span<tvec3<f32>> colors);

template void write_point_cloud_ply(const std::string& filepath, const cpu::span<tvec3<f32>> points, const cpu::span<tvec3<u8>> colors);
template void write_point_cloud_ply(const std::string& filepath, const cpu::span<tvec3<f32>> points, const cpu::span<tvec3<u16>> colors);
template void write_point_cloud_ply(const std::string& filepath, const cpu::span<tvec3<f32>> points, const cpu::span<tvec3<u32>> colors);

template void write_point_cloud_ply(const std::string& filepath, const cpu::span<tvec3<f32>> points, const cpu::span<tvec3<i32>> colors);

}