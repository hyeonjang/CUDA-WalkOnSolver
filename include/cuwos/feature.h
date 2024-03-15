#pragma once

#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

#include <cuwos/memory.h>

//https://stackoverflow.com/questions/50814669/element-wise-tuple-addition
#define EWISE_TUPLE(operation, symbol, count) \
template <std::size_t I, typename ... Args> \
DEVICE auto operation_impl_##count(const thrust::tuple<Args...>& a, const thrust::tuple<Args...>& b) { \
    if constexpr (I < thrust::tuple_size<thrust::tuple<Args...>>::value) \
        return thrust::get<I>(a) symbol thrust::get<I>(b); \
    else return thrust::null_type(); \
} \
template <std::size_t... Is, typename... Args> \
DEVICE thrust::tuple<Args...> operation_impl_##count(std::index_sequence<Is...>, const thrust::tuple<Args...>& a, const thrust::tuple<Args...>& b) { \
    return thrust::make_tuple(operation_impl_##count<Is>(a, b)...); \
} \
template <typename... Args> \
DEVICE thrust::tuple<Args...> operation(const thrust::tuple<Args...>& a, const thrust::tuple<Args...>& b) { \
    return operation_impl_##count(std::index_sequence_for<Args...>(), a, b); \
} \
\
template <typename Constant, std::size_t I, typename... Args> \
DEVICE auto operation_impl_scalar##count(const thrust::tuple<Args...>& a, Constant b) { \
    if constexpr (I < thrust::tuple_size<thrust::tuple<Args...>>::value) \
        return thrust::get<I>(a) symbol b; \
    else return thrust::null_type(); \
} \
template <typename Constant, std::size_t... Is, typename... Args> \
DEVICE thrust::tuple<Args...> operation_impl_scalar##count(std::index_sequence<Is...>, const thrust::tuple<Args...>& a, Constant b) { \
    return thrust::make_tuple(operation_impl_scalar##count<Constant, Is>(a, b)...); \
} \
template <typename Constant, typename... Args> \
DEVICE thrust::tuple<Args...> operation(const thrust::tuple<Args...>& a, Constant b) { \
    return operation_impl_scalar##count(std::index_sequence_for<Args...>(), a, b); \
}

EWISE_TUPLE(operator+, +, sum);
EWISE_TUPLE(operator/, /, div);

// trait to get variadic
template <size_t I, typename... K>
struct Nth {
    using type = std::tuple_element_t<I, std::tuple<K...>>;
    static constexpr size_t size = sizeof(type);
};

template <typename... T>
class Features {
public:
    using Type = thrust::tuple<T ...>;
    using Data = thrust::tuple<T* ...>;
    using Iter = thrust::zip_iterator<Data>;
    static constexpr size_t Size = sizeof...(T);

    Features(size_t _size) :m_size(_size) {
        const size_t padd = (m_size + 3)/4 * 4;

        m_allocation = std::make_shared<gpu::memory<u8>>(
            padd * (sizeof(T) + ...)
        );

        m_data = assign_pointers(std::index_sequence_for<T...>(), padd);
    }

    template <typename Constant>
    __forceinline__
    static HOST_DEVICE Type init(Constant c) {
        return thrust::make_tuple(T(c)...);
    }

    __forceinline__
    static HOST_DEVICE Type zero() {
        return thrust::make_tuple(T(0.0)...);
    }

    __forceinline__
    static HOST_DEVICE Type make(T... args) {
        return thrust::make_tuple(args...);
    }

    Data data() const {
        return m_data;
    }

    Iter iter() const {
        return thrust::make_zip_iterator(m_data);
    }

    size_t size() const {
        return m_size;
    }

private:
    template<size_t... Is>
    auto assign_pointers(std::index_sequence<Is...>, size_t size) -> thrust::tuple<T*...> {
        // padding for consistency

        // sequential offset
        size_t offset[Size+1]{0, ((Nth<Is, T...>::size * size), ...) };

        // to thrust tuple
        return thrust::make_tuple(assign_pointers<Is>(offset[Is], size)...);
    }

    template<size_t I>
    auto assign_pointers(size_t offset, size_t size) -> typename Nth<I, T*...>::type {
        return reinterpret_cast<typename Nth<I, T*...>::type>(m_allocation->data() + offset);
    }

private:
    Data m_data;
    size_t m_size;
    
    // memory manager
    std::shared_ptr<gpu::memory<u8>> m_allocation;
};
