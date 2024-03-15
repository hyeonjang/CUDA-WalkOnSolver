#include <cuwos/common.h>

int cuda_runtime_version() {
    int version;
    CHECK_CUDA(cudaRuntimeGetVersion(&version));
    return version;
}

int cuda_device() {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    return device;
}

void set_cuda_device(int device) {
    CHECK_CUDA(cudaSetDevice(device));
}

bool cuda_supports_virtual_memory(int device) {
    int supports_vmm;
	CHECK_CUDA(cuDeviceGetAttribute(&supports_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, device));
	return supports_vmm != 0;
};

size_t cuda_memory_granularity(int device) {
	size_t granularity;
	CUmemAllocationProp prop = {};
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = 0;
	CUresult granularity_result = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
	if (granularity_result == CUDA_ERROR_NOT_SUPPORTED) {
		return 1;
	}
	// cu_throw(granularity_result);
	return granularity;
}

MemoryInfo cuda_memory_info() {
    MemoryInfo info;
    CHECK_CUDA(cudaMemGetInfo(&info.free, &info.total));
    info.used = info.total - info.free;
    return info;
};