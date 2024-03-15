#pragma once
#include <cuwos/vector.h>

using namespace tcnn;

template <typename T, size_t N>
struct BaseAABB {

    using Tvec = tvec<T, N>;

    Tvec lower = Tvec(std::numeric_limits<T>::max());
    Tvec upper = Tvec(std::numeric_limits<T>::min());

    __forceinline__
    HOST_DEVICE Tvec centroid() const {
        return (lower + upper) * static_cast<T>(0.5);
    }

    // instant-ngp
    __forceinline__
    HOST_DEVICE T shortest_distance(Tvec p) const {
        return length2(max(min(lower - p, p - upper), Tvec(0.0f)));
    }

    __forceinline__
    HOST_DEVICE T longest_distance(Tvec p) const {
        return length2(max(max(lower - p, p - upper), Tvec(0.0f)));
    }

    __forceinline__
    HOST_DEVICE T size() const {
        return length2(upper - lower);
    } 
};

template <typename T, size_t N> __forceinline__
HOST_DEVICE BaseAABB<T, N> merge(const BaseAABB<T, N>& lhs, const BaseAABB<T, N>& rhs) {
    return BaseAABB<T, N>{min(lhs.lower, rhs.lower), max(lhs.upper, rhs.upper)};
}

template <typename T, size_t N> __forceinline__
DEVICE void atomic_merge(BaseAABB<T, N>* lhs, const BaseAABB<T, N>& rhs);



