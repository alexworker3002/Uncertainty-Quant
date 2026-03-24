#include "dmt_implicit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace dmt_implicit {

namespace {

inline std::size_t flat_index(std::size_t y, std::size_t x, std::size_t w) {
  return y * w + x;
}

inline bool has_dim(const std::vector<int>& dims, int target) {
  for (int d : dims) {
    if (d == target) {
      return true;
    }
  }
  return false;
}

inline double clamp01(double v) {
  return std::min(1.0, std::max(0.0, v));
}

struct BitFlags {
  explicit BitFlags(std::size_t n_bits) : words((n_bits + 63U) / 64U, 0ULL) {}

  bool get(std::size_t i) const {
    return (words[i >> 6U] >> (i & 63U)) & 1ULL;
  }

  void set(std::size_t i) {
    words[i >> 6U] |= (1ULL << (i & 63U));
  }

  std::vector<std::uint64_t> words;
};

} // namespace

PersistenceOutput extract_persistence_2d(
    const double* prob,
    std::size_t h,
    std::size_t w,
    double min_persistence,
    const std::vector<int>& homology_dims) {
  PersistenceOutput out;

  if (prob == nullptr || h == 0 || w == 0) {
    return out;
  }

  const std::size_t n = h * w;
  std::vector<double> pvals(n, 0.0);
  out.filtration_values.resize(n);

  for (std::size_t y = 0; y < h; ++y) {
    for (std::size_t x = 0; x < w; ++x) {
      const std::size_t idx = flat_index(y, x, w);
      const double p = clamp01(prob[idx]);
      pvals[idx] = p;
      out.filtration_values[idx] = 1.0 - p;
    }
  }

  // Bit-array state (prototype): marks cells already consumed by a selected pair.
  BitFlags paired_flags(n);

  if (has_dim(homology_dims, 0)) {
    constexpr std::int64_t kInvalidIndex = -1;
    const int dy[4] = {-1, 1, 0, 0};
    const int dx[4] = {0, 0, -1, 1};

    for (std::size_t y = 0; y < h; ++y) {
      for (std::size_t x = 0; x < w; ++x) {
        const std::size_t idx = flat_index(y, x, w);
        if (paired_flags.get(idx)) {
          continue;
        }

        const double center = pvals[idx];
        bool is_local_max = true;
        double min_neighbor = std::numeric_limits<double>::infinity();
        std::int64_t min_neighbor_idx = kInvalidIndex;

        for (int k = 0; k < 4; ++k) {
          const int ny = static_cast<int>(y) + dy[k];
          const int nx = static_cast<int>(x) + dx[k];
          if (ny < 0 || nx < 0 || ny >= static_cast<int>(h) || nx >= static_cast<int>(w)) {
            continue;
          }
          const std::size_t nidx = flat_index(static_cast<std::size_t>(ny), static_cast<std::size_t>(nx), w);
          const double np = pvals[nidx];

          if (np >= center) {
            is_local_max = false;
          }
          if (!paired_flags.get(nidx) && np < min_neighbor) {
            min_neighbor = np;
            min_neighbor_idx = static_cast<std::int64_t>(nidx);
          }
        }

        if (!is_local_max) {
          continue;
        }

        const double birth = center;
        const double death = std::isfinite(min_neighbor) ? min_neighbor : 0.0;
        const double persistence = birth - death;
        if (persistence < min_persistence) {
          continue;
        }

        out.pairs.push_back(birth);
        out.pairs.push_back(death);
        out.dimensions.push_back(0);
        out.birth_indices.push_back(static_cast<std::int64_t>(idx));
        out.death_indices.push_back(min_neighbor_idx);

        paired_flags.set(idx);
        if (min_neighbor_idx >= 0) {
          paired_flags.set(static_cast<std::size_t>(min_neighbor_idx));
        }
      }
    }
  }

  // Initial H1 proxy (prototype): detect high-valued pixel rings in 3x3 neighborhood.
  // We add small-weight cycles with synthetic death = local minimum around ring.
  if (has_dim(homology_dims, 1) && h >= 3 && w >= 3) {
    const double ring_thresh = 0.60;

    for (std::size_t y = 1; y + 1 < h; ++y) {
      for (std::size_t x = 1; x + 1 < w; ++x) {
        const std::size_t c = flat_index(y, x, w);
        if (paired_flags.get(c)) {
          continue;
        }

        // 8-neighborhood ring (clockwise)
        const std::size_t ring[8] = {
            flat_index(y - 1, x - 1, w), flat_index(y - 1, x, w), flat_index(y - 1, x + 1, w),
            flat_index(y, x + 1, w),     flat_index(y + 1, x + 1, w), flat_index(y + 1, x, w),
            flat_index(y + 1, x - 1, w), flat_index(y, x - 1, w)};

        bool ring_ok = true;
        double birth = std::numeric_limits<double>::infinity();
        double death = 1.0;
        std::size_t birth_idx = ring[0];

        for (std::size_t ridx : ring) {
          const double v = pvals[ridx];
          if (v < ring_thresh || paired_flags.get(ridx)) {
            ring_ok = false;
            break;
          }
          if (v < birth) {
            birth = v;
            birth_idx = ridx;
          }
        }

        if (!ring_ok) {
          continue;
        }

        // center should be lower to mimic a hole.
        const double center = pvals[c];
        death = std::min(center, birth);

        const double persistence = birth - death;
        if (persistence < min_persistence) {
          continue;
        }

        out.pairs.push_back(birth);
        out.pairs.push_back(death);
        out.dimensions.push_back(1);
        out.birth_indices.push_back(static_cast<std::int64_t>(birth_idx));
        out.death_indices.push_back(static_cast<std::int64_t>(c));

        paired_flags.set(c);
        paired_flags.set(birth_idx);
      }
    }
  }

  return out;
}

} // namespace dmt_implicit
