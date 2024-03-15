/*
 * Modifications made by Hyeonjang An
 * Date of Modifications: January 12, 2024
 * 
 * Changes made:
 * 	runtime check error
 * 	type alias like rust
 *
 * The original copyright and license text as provided by NVIDIA CORPORATION is retained below.
 *
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   common.h
 * Original authors: Thomas Müller and Nikolaus Binder, NVIDIA
 * Modified by: Hyeonjang An
 */

#pragma once

#include <array>
#include <string>
#include <functional>
#include <sstream>

#ifndef CGK_OPTIX
#include <fmt/format.h>
#endif

#if defined(__CUDACC__)
    #include <cuda.h>
    #include <driver_types.h>
    #include <cuda_fp16.h>
	#include <cusparse.h>
	// #include <cuco/static_map.cuh>
#endif

#if defined(__WIN32__)
	#define _USE_MATH_DEFINES
	#include <cmath>
#endif

#ifndef M_PI
	#define M_PI 3.1415926535897923846
#endif
#define M_f_PI (f32)M_PI

#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
#define HOST_DEVICE __host__ __device__
#define DEVICE __device__
#define HOST __host__
#else
#define HOST_DEVICE
#define DEVICE
#define HOST
#endif

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)

// 
using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using f32 = float;
using f64 = double;

template <typename T> struct INVALID;
#define DEFINE_INVALID(type) \
	template <> struct INVALID<type> { static constexpr type value = std::numeric_limits<type>::quiet_NaN(); };
DEFINE_INVALID(i8)
DEFINE_INVALID(i16)
DEFINE_INVALID(i32)
DEFINE_INVALID(i64)
DEFINE_INVALID(u8)
DEFINE_INVALID(u16)
DEFINE_INVALID(u32)
DEFINE_INVALID(u64)
DEFINE_INVALID(f32)
DEFINE_INVALID(f64)

enum class LogSeverity {
	Info,
	Debug,
	Warning,
	Error,
	Success,
};

const std::function<void(LogSeverity, const std::string&)>& log_callback();
void set_log_callback(const std::function<void(LogSeverity, const std::string&)>& callback);

template <typename... Ts>
void log(LogSeverity severity, const std::string& msg, Ts&&... args) {
	// log_callback()(severity, fmt::format(msg, std::forward<Ts>(args)...));
}

template <typename... Ts> void log_info(const std::string& msg, Ts&&... args) { log(LogSeverity::Info, msg, std::forward<Ts>(args)...); }
template <typename... Ts> void log_debug(const std::string& msg, Ts&&... args) { log(LogSeverity::Debug, msg, std::forward<Ts>(args)...); }
template <typename... Ts> void log_warning(const std::string& msg, Ts&&... args) { log(LogSeverity::Warning, msg, std::forward<Ts>(args)...); }
template <typename... Ts> void log_error(const std::string& msg, Ts&&... args) { log(LogSeverity::Error, msg, std::forward<Ts>(args)...); }
template <typename... Ts> void log_success(const std::string& msg, Ts&&... args) { log(LogSeverity::Success, msg, std::forward<Ts>(args)...); }

namespace check {
#define CHECK(some, ...) { \
    do {  \
        if (!(some))  \
            throw std::runtime_error{fmt::format(FILE_LINE " failed: " __VA_ARGS__)}; \
    } while(0); \
}

template <typename T>
const char* cuda_get_error(T error) {
	if constexpr (std::is_same<T, cudaError_t>::value) {
		return cudaGetErrorString(error);
	} else if constexpr(std::is_same<T, CUresult>::value) {
		const char* msg;
		cuGetErrorName(error, &msg);
		return msg;
	} else if constexpr(std::is_same<T, cusparseStatus_t>::value) {
		return cusparseGetErrorString(error);
	}
	return "unspecified cuda error";
}

// auto success = static_cast<decltype(result)>(0);
#define CHECK_CUDA(func) {\
	auto result = (func); \
	if (result != static_cast<decltype(result)>(0)) \
		throw std::runtime_error{fmt::format(FILE_LINE " cuda failed {}: {}", (int)result, cuda_get_error(result))}; \
}
} // namespace check


using namespace check;

int cuda_runtime_version();
int cuda_device();
void set_cuda_device(int device);
bool cuda_supports_virtual_memory(int device);
inline bool cuda_supports_virtual_memory() {
	return cuda_supports_virtual_memory(cuda_device());
}

size_t cuda_memory_granularity(int device);
inline size_t cuda_memory_granularity() {
	return cuda_memory_granularity(cuda_device());
}
struct MemoryInfo {
    size_t total;
    size_t free;
    size_t used;
};
MemoryInfo cuda_memory_info();
 
 inline std::string bytes_to_string(size_t bytes) {
	std::array<std::string, 7> suffixes = {{ "B", "KB", "MB", "GB", "TB", "PB", "EB" }};

	double count = (double)bytes;
	uint32_t i = 0;
	for (; i < suffixes.size() && count >= 1024; ++i) {
		count /= 1024;
	}

	std::ostringstream oss;
	oss.precision(3);
	oss << count << " " << suffixes[i];
	return oss.str();
}

#if defined(__CUDA_ARCH__)
	#define TCNN_PRAGMA_UNROLL _Pragma("unroll")
	#define TCNN_PRAGMA_NO_UNROLL _Pragma("unroll 1")
#else
	#define TCNN_PRAGMA_UNROLL
	#define TCNN_PRAGMA_NO_UNROLL
#endif

#ifdef __CUDACC__
#  ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#    pragma nv_diag_suppress = unsigned_compare_with_zero
#  else
#    pragma diag_suppress = unsigned_compare_with_zero
#  endif
#endif

#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
#define TCNN_HOST_DEVICE __host__ __device__
#define TCNN_DEVICE __device__
#define TCNN_HOST __host__
#else
#define TCNN_HOST_DEVICE
#define TCNN_DEVICE
#define TCNN_HOST
#endif

constexpr uint32_t BATCH_SIZE_GRANULARITY = 256;
constexpr uint32_t N_THREADS_LINEAR = 128;
constexpr uint32_t WARP_SIZE = 32;

// Lower-case constants kept for backward compatibility with user code.
constexpr uint32_t batch_size_granularity = BATCH_SIZE_GRANULARITY;
constexpr uint32_t n_threads_linear = N_THREADS_LINEAR;

template <typename T>
TCNN_HOST_DEVICE T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

template <typename T>
TCNN_HOST_DEVICE T next_multiple(T val, T divisor) {
	return div_round_up(val, divisor) * divisor;
}

template <typename T>
TCNN_HOST_DEVICE T previous_multiple(T val, T divisor) {
	return (val / divisor) * divisor;
}

template <typename T>
constexpr TCNN_HOST_DEVICE uint32_t n_blocks_linear(T n_elements, uint32_t n_threads = N_THREADS_LINEAR) {
	return (uint32_t)div_round_up(n_elements, (T)n_threads);
}

#if defined(__CUDACC__) || (defined(__clang__) && defined(__CUDA__))
template <typename K, typename T, typename ... Types>
inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types ... args) {
	if (n_elements <= 0) {
		return;
	}
	kernel<<<n_blocks_linear(n_elements), N_THREADS_LINEAR, shmem_size, stream>>>(n_elements, args...);
}
#endif