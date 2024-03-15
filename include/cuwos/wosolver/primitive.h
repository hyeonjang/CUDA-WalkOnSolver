#pragma once

#include <thrust/tuple.h>
#include <pcg32.h>
#include <cuwos/memory.h>
#include <cuwos/vector.h>

template <typename T, size_t N>
struct Primitive {
    static constexpr size_t dim = N;
    using AABB = BaseAABB<T, dim>;
};

struct Triangle : Primitive<f32, 3> {
    vec3 v0, v1, v2;
    vec3 n; // better to be cached? don'y know ...

    HOST_DEVICE Triangle(){}

    HOST_DEVICE Triangle(vec3 a, vec3 b, vec3 c)
    : v0(a), v1(b), v2(c), n(cross(b - a, a - c)) {}

    HOST_DEVICE Triangle(const Triangle& other) {
        v0 = other.v0;
        v1 = other.v1;
        v2 = other.v2;
        n = other.n;
    }

    HOST_DEVICE Triangle& operator=(const Triangle& other) {
        v0 = other.v0;
        v1 = other.v1;
        v2 = other.v2;
        n = other.n;
        return *this;
    }

    HOST_DEVICE  AABB aabb() const {
        return AABB{min(min(v0, v1), v2), max(max(v0, v1), v2)};
    }

    // instant-ngp
	// based on https://www.iquilezles.org/articles/distfunctions/
    HOST_DEVICE inline float distance_square(vec3 p) const {
        vec3 v10 = v1 - v0; vec3 p0 = p - v0;
        vec3 v21 = v2 - v1; vec3 p1 = p - v1;
        vec3 v02 = v0 - v2; vec3 p2 = p - v2;
        vec3 nor = cross( v10, v02 );

		return
			// inside/outside test
			(sign(dot(cross(v10, nor), p0)) + 
             sign(dot(cross(v21, nor), p1)) + 
             sign(dot(cross(v02, nor), p2)) < 2.0f)
			?
			// 3 edges
			std::min({
				length2(v10 * tcnn::clamp(dot(v10, p0) / length2(v10), 0.0f, 1.0f)-p0),
				length2(v21 * tcnn::clamp(dot(v21, p1) / length2(v21), 0.0f, 1.0f)-p1),
				length2(v02 * tcnn::clamp(dot(v02, p2) / length2(v02), 0.0f, 1.0f)-p2),
			})
			:
			// 1 face
			dot(nor, p0) * dot(nor, p0) / length2(nor);
    }

    HOST_DEVICE inline thrust::tuple<f32, bool> signed_distance_square(vec3 p) const {
        vec3 v10 = v1 - v0; vec3 p0 = p - v0;
        vec3 v21 = v2 - v1; vec3 p1 = p - v1;
        vec3 v02 = v0 - v2; vec3 p2 = p - v2;
        vec3 nor = cross( v10, v02 );

        f32 dotted = dot(normalize(cross(v10, v2 - v0)), p0);
		return {
			// inside/outside test
			((sign(dot(cross(v10, nor), p0)) + 
             sign(dot(cross(v21, nor), p1)) + 
             sign(dot(cross(v02, nor), p2)) < 2.0f)
			?
			// 3 edges
			std::min({
				length2(v10 * tcnn::clamp(dot(v10, p0) / length2(v10), 0.0f, 1.0f)-p0),
				length2(v21 * tcnn::clamp(dot(v21, p1) / length2(v21), 0.0f, 1.0f)-p1),
				length2(v02 * tcnn::clamp(dot(v02, p2) / length2(v02), 0.0f, 1.0f)-p2),
			})
			:
			// 1 face
			dot(nor, p0) * dot(nor, p0) / length2(nor)), dotted < 0.0f
        };
    } 

    // inigo quilez: https://www.shadertoy.com/view/ttfGWl
    HOST_DEVICE __forceinline__ vec3 closest_point(vec3 p) const {
        vec3 v10 = v1 - v0; vec3 p0 = p - v0;
        vec3 v21 = v2 - v1; vec3 p1 = p - v1;
        vec3 v02 = v0 - v2; vec3 p2 = p - v2;
        vec3 nor = cross( v10, v02 );

        // method 1, in 3D space
        vec3  q = cross( nor, p0 );
        float d = 1.0/length2(nor);
        float u = d*dot( q, v02 );
        float v = d*dot( q, v10 );
        float w = 1.0-u-v;
        
        if( u<0.0 ) { w = clamp( dot(p2,v02)/length2(v02), 0.0f, 1.0f ); u = 0.0; v = 1.0-w; }
        else if( v<0.0 ) { u = clamp( dot(p0,v10)/length2(v10), 0.0f, 1.0f ); v = 0.0; w = 1.0-u; }
        else if( w<0.0 ) { v = clamp( dot(p1,v21)/length2(v21), 0.0f, 1.0f ); w = 0.0; u = 1.0-v; }
        
        return u*v1 + v*v2 + w*v0;
    }

    HOST_DEVICE __forceinline__ std::pair<vec3, float> closest_point_distance(vec3 p) const {
        vec3 point = closest_point(p);
        return {point, length2(p-point)};
    }

    HOST_DEVICE __forceinline__ bool is_inner_point(vec3 p) const {
        		// Move the triangle so that the point becomes the
		// triangles origin
		vec3 local_a = v0 - p;
		vec3 local_b = v1 - p;
		vec3 local_c = v2 - p;

		// The point should be moved too, so they are both
		// relative, but because we don't use p in the
		// equation anymore, we don't need it!
		// p -= p;

		// Compute the normal vectors for triangles:
		// u = normal of PBC
		// v = normal of PCA
		// w = normal of PAB

		vec3 u = cross(local_b, local_c);
		vec3 v = cross(local_c, local_a);
		vec3 w = cross(local_a, local_b);

		// Test to see if the normals are facing the same direction.
		// If yes, the point is inside, otherwise it isn't.
		return dot(u, v) >= 0.0f && dot(u, w) >= 0.0f;
    }
};

__forceinline__
DEVICE vec3 sample_sphere(vec3 center, f32 radius, f32 gen) {    
    f32 theta = 2.0f * M_PI * gen;          // Azimuthal angle
    f32 phi = std::acos(1.0f - 2.0f * gen); // Polar angle
    return vec3(center.x + radius * sin(phi)*cos(theta),
                center.y + radius * sin(phi)*sin(theta), 
                center.z + radius * cos(phi));
}

struct Sphere : Primitive<f32, 3> {
    vec3 p; f32 r;

    HOST_DEVICE Sphere(vec3 _p, f32 _r): p(_p), r(_r) {};

    HOST_DEVICE vec3 sample(f32 radius, pcg32* rng) const {
        f32 theta = 2.0 * M_PI * rng->next_float();
        f32 phi = std::acos(-2.0f* rng->next_float() + 1.0f);
        return vec3(p.x + radius * sin(phi)*cos(theta),
                    p.y + radius * sin(phi)*sin(theta), 
                    p.z + radius * cos(phi));
    }

    HOST_DEVICE vec3 sample_surface(pcg32* rng) const {
        return sample(this->r, rng);
    }

    HOST_DEVICE thrust::pair<vec3, f32> sample_volume(pcg32* rng) const {
        f32 ir = r * rng->next_float();
        return { sample(ir, rng), ir };
    }
};