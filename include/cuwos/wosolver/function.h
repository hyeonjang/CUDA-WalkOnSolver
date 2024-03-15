#pragma once

#include <nvfunctional>

#include <cuwos/common.h>
#include <cuwos/vector.h>
#include <cuwos/wosolver/primitive.h>

struct GreensFunctionBall {
    HOST_DEVICE f32 operator()(f32 r, f32 R) const {
        return (1.0f/(r - R))/(4.0f*M_PI);
    }

    HOST_DEVICE vec3 gradient(vec3 center, f32 R, vec3 y, f32 r) const {
        return (y - center) * ((1.0f/(r*r*r - R*R*R))/(4.0f*M_f_PI));
    }
};

// template <typename Value>
template <typename Derived, typename _BoundaryCondition>
struct PDE {
    using BoundaryCondition = _BoundaryCondition;
    using Value = decltype(std::declval<BoundaryCondition>()(std::declval<vec3>()));

    HOST_DEVICE PDE(): g(BoundaryCondition()) {}
    HOST_DEVICE PDE(BoundaryCondition _g): g(_g) {}

    HOST_DEVICE Value operator()(vec3 center, f32 R, vec3 y, f32 r) {
        return static_cast<Derived*>(this)->operator()(center, R, y, r);
    }

    HOST_DEVICE Value gradient(vec3 center, f32 R, vec3 y, f32 r) {
        return static_cast<Derived*>(this)->gradient(center, R, y, r);
    }
    BoundaryCondition g;
};

template <typename BoundaryCondition>
struct Laplace : public PDE<Laplace<BoundaryCondition>, BoundaryCondition> {
    using Base = PDE<Laplace<BoundaryCondition>, BoundaryCondition>;
    using Base::Value;
    using Base::g;

    HOST_DEVICE Laplace(): Base() {}
    HOST_DEVICE Laplace(BoundaryCondition g): Base(g) {}

    HOST_DEVICE virtual typename Base::Value operator()(vec3 center, f32 R, vec3 y, f32 r) {
        return typename Base::Value(0.0f);
    }
};

template <typename BoundaryCondition, typename SourceTerm>
struct Poisson : public PDE<Poisson<BoundaryCondition, SourceTerm>, BoundaryCondition> {
    using Base = PDE<Poisson<BoundaryCondition, SourceTerm>, BoundaryCondition>;
    using Base::Value;
    using Base::g;

    HOST_DEVICE Poisson(): G(GreensFunctionBall()) {}

    HOST_DEVICE typename Base::Value operator()(vec3 center, f32 R, vec3 y, f32 r) {
        return R * R * f(y) * G(r, R);
    }

    HOST_DEVICE vec3 gradient(vec3 center, f32 R, vec3 y, f32 r) {
        return f(y) * G.gradient(center, R, y, r);
    }

    SourceTerm f;
    GreensFunctionBall G;
};

// struct Diffusion