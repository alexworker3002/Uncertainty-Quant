#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dmt_implicit {

struct PersistenceOutput {
  std::vector<double> pairs;            // flattened (M, 2): [b0, d0, b1, d1, ...]
  std::vector<std::int64_t> dimensions; // (M,)
  std::vector<std::int64_t> birth_indices;
  std::vector<std::int64_t> death_indices;
  std::vector<double> filtration_values; // compact output for diagnostics
};

// -----------------------------
// Stage-1/2 foundational structs (2D)
// -----------------------------
struct Filtration2D {
  std::size_t h{0};
  std::size_t w{0};

  std::vector<double> vertex; // size h*w
  std::vector<double> edge_h; // size h*(w-1)
  std::vector<double> edge_v; // size (h-1)*w
  std::vector<double> face;   // size (h-1)*(w-1)

  std::vector<double> vertex_raw; // original clamped probability
};

struct GradientField2D {
  std::vector<std::int64_t> v2e; // size |V|, points to global edge id
  std::vector<std::int64_t> e2v; // size |E|, points to global vertex id
  std::vector<std::int64_t> e2f; // size |E|, points to face id
  std::vector<std::int64_t> f2e; // size |F|, points to global edge id
};

// -----------------------------
// Stage-1/2 foundational structs (3D scaffold)
// -----------------------------
struct Filtration3D {
  std::size_t d{0};
  std::size_t h{0};
  std::size_t w{0};

  std::vector<double> vertex;   // D*H*W
  std::vector<double> edge_x;   // D*H*(W-1)
  std::vector<double> edge_y;   // D*(H-1)*W
  std::vector<double> edge_z;   // (D-1)*H*W
  std::vector<double> face_xy;  // D*(H-1)*(W-1)
  std::vector<double> face_xz;  // (D-1)*H*(W-1)
  std::vector<double> face_yz;  // (D-1)*(H-1)*W
  std::vector<double> cube;     // (D-1)*(H-1)*(W-1)

  std::vector<double> vertex_raw;
};

struct GradientField3D {
  std::vector<std::int64_t> v2e;
  std::vector<std::int64_t> e2v;
  std::vector<std::int64_t> e2f;
  std::vector<std::int64_t> f2e;
  std::vector<std::int64_t> f2c;
  std::vector<std::int64_t> c2f;
};

Filtration2D build_filtration_2d_upper_star(
    const double* prob,
    std::size_t h,
    std::size_t w,
    double tie_eps = 1e-12);

GradientField2D build_gradient_field_2d_robins(const Filtration2D& fil);

Filtration3D build_filtration_3d_upper_star(
    const double* prob,
    std::size_t d,
    std::size_t h,
    std::size_t w,
    double tie_eps = 1e-12);

GradientField3D build_gradient_field_3d_scaffold(const Filtration3D& fil);

PersistenceOutput extract_persistence_2d(
    const double* prob,
    std::size_t h,
    std::size_t w,
    double min_persistence,
    const std::vector<int>& homology_dims);

} // namespace dmt_implicit
