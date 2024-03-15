#include <cuwos/wosolver/solver.h>

template class WoSolver<TriangleLBVH>;

// for binding
template WalkFeatures<Laplace<Sine>> WoSolver<TriangleLBVH>::estimate_features<Laplace<Sine>>(cudaStream_t, gpu::span<vec3>, Laplace<Sine>&) const;
