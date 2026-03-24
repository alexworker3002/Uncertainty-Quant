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

PersistenceOutput extract_persistence_2d(
    const double* prob,
    std::size_t h,
    std::size_t w,
    double min_persistence,
    const std::vector<int>& homology_dims);

} // namespace dmt_implicit
