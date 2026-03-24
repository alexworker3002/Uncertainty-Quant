#include "dmt_implicit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace dmt_implicit {

namespace {

inline std::size_t vid2(std::size_t y, std::size_t x, std::size_t w) {
  return y * w + x;
}

inline std::size_t ehid2(std::size_t y, std::size_t x, std::size_t w) {
  return y * (w - 1) + x;
}

inline std::size_t fid2(std::size_t y, std::size_t x, std::size_t w) {
  return y * (w - 1) + x;
}

inline std::size_t vid3(std::size_t z, std::size_t y, std::size_t x, std::size_t h, std::size_t w) {
  return (z * h + y) * w + x;
}

inline double clamp01(double v) {
  return std::min(1.0, std::max(0.0, v));
}

inline double perturb(double v, std::size_t global_id, double eps) {
  return v - eps * static_cast<double>(global_id + 1);
}

struct DSU {
  explicit DSU(std::size_t n) : parent(n, -1), rank(n, 0), birth_val(n, -1.0), birth_vertex(n, -1) {}

  std::int64_t find(std::int64_t x) {
    if (parent[static_cast<std::size_t>(x)] == x) {
      return x;
    }
    parent[static_cast<std::size_t>(x)] = find(parent[static_cast<std::size_t>(x)]);
    return parent[static_cast<std::size_t>(x)];
  }

  void activate(std::int64_t x, double b, std::int64_t bvid) {
    const auto i = static_cast<std::size_t>(x);
    parent[i] = x;
    rank[i] = 0;
    birth_val[i] = b;
    birth_vertex[i] = bvid;
  }

  bool active(std::int64_t x) const {
    return parent[static_cast<std::size_t>(x)] >= 0;
  }

  std::vector<std::int64_t> parent;
  std::vector<std::uint8_t> rank;
  std::vector<double> birth_val;
  std::vector<std::int64_t> birth_vertex;
};

} // namespace

Filtration2D build_filtration_2d_upper_star(
    const double* prob,
    std::size_t h,
    std::size_t w,
    double tie_eps) {
  Filtration2D out;
  out.h = h;
  out.w = w;

  if (prob == nullptr || h == 0 || w == 0) {
    return out;
  }

  const std::size_t n_v = h * w;
  const std::size_t n_eh = (w >= 2) ? h * (w - 1) : 0;
  const std::size_t n_ev = (h >= 2) ? (h - 1) * w : 0;
  const std::size_t n_f = (h >= 2 && w >= 2) ? (h - 1) * (w - 1) : 0;

  out.vertex.resize(n_v);
  out.vertex_raw.resize(n_v);
  out.edge_h.resize(n_eh);
  out.edge_v.resize(n_ev);
  out.face.resize(n_f);

  const std::size_t base_v = 0;
  const std::size_t base_eh = base_v + n_v;
  const std::size_t base_ev = base_eh + n_eh;
  const std::size_t base_f = base_ev + n_ev;

  for (std::size_t y = 0; y < h; ++y) {
    for (std::size_t x = 0; x < w; ++x) {
      const std::size_t i = vid2(y, x, w);
      const double p = clamp01(prob[i]);
      out.vertex_raw[i] = p;
      out.vertex[i] = perturb(p, base_v + i, tie_eps);
    }
  }

  for (std::size_t y = 0; y < h; ++y) {
    for (std::size_t x = 0; x + 1 < w; ++x) {
      const std::size_t e = ehid2(y, x, w);
      const double p0 = out.vertex_raw[vid2(y, x, w)];
      const double p1 = out.vertex_raw[vid2(y, x + 1, w)];
      out.edge_h[e] = perturb(std::min(p0, p1), base_eh + e, tie_eps);
    }
  }

  for (std::size_t y = 0; y + 1 < h; ++y) {
    for (std::size_t x = 0; x < w; ++x) {
      const std::size_t e = y * w + x;
      const double p0 = out.vertex_raw[vid2(y, x, w)];
      const double p1 = out.vertex_raw[vid2(y + 1, x, w)];
      out.edge_v[e] = perturb(std::min(p0, p1), base_ev + e, tie_eps);
    }
  }

  for (std::size_t y = 0; y + 1 < h; ++y) {
    for (std::size_t x = 0; x + 1 < w; ++x) {
      const std::size_t f = fid2(y, x, w);
      const double p00 = out.vertex_raw[vid2(y, x, w)];
      const double p01 = out.vertex_raw[vid2(y, x + 1, w)];
      const double p10 = out.vertex_raw[vid2(y + 1, x, w)];
      const double p11 = out.vertex_raw[vid2(y + 1, x + 1, w)];
      const double v = std::min(std::min(p00, p01), std::min(p10, p11));
      out.face[f] = perturb(v, base_f + f, tie_eps);
    }
  }

  return out;
}

GradientField2D build_gradient_field_2d_robins(const Filtration2D& fil) {
  GradientField2D g;

  const std::size_t h = fil.h;
  const std::size_t w = fil.w;
  const std::size_t n_v = h * w;
  const std::size_t n_eh = (w >= 2) ? h * (w - 1) : 0;
  const std::size_t n_ev = (h >= 2) ? (h - 1) * w : 0;
  const std::size_t n_e = n_eh + n_ev;
  const std::size_t n_f = (h >= 2 && w >= 2) ? (h - 1) * (w - 1) : 0;

  g.v2e.assign(n_v, -1);
  g.e2v.assign(n_e, -1);
  g.e2f.assign(n_e, -1);
  g.f2e.assign(n_f, -1);

  struct CellRef {
    int dim;            // 0 vertex, 1 edge, 2 face
    std::size_t id;     // local id in dimension block
    double key;         // strictly ordered filtration key
    std::size_t global; // global tiebreak id
  };

  std::vector<CellRef> cells;
  cells.reserve(n_v + n_e + n_f);

  for (std::size_t v = 0; v < n_v; ++v) {
    cells.push_back(CellRef{0, v, fil.vertex[v], v});
  }
  for (std::size_t e = 0; e < n_e; ++e) {
    const double k = (e < n_eh) ? fil.edge_h[e] : fil.edge_v[e - n_eh];
    cells.push_back(CellRef{1, e, k, n_v + e});
  }
  for (std::size_t f = 0; f < n_f; ++f) {
    cells.push_back(CellRef{2, f, fil.face[f], n_v + n_e + f});
  }

  std::sort(cells.begin(), cells.end(), [](const CellRef& a, const CellRef& b) {
    if (a.key != b.key) {
      return a.key > b.key; // upper-star: larger comes first
    }
    return a.global > b.global; // strict total ordering fallback
  });

  auto edge_endpoints = [&](std::size_t e, std::size_t& v0, std::size_t& v1) {
    if (e < n_eh) {
      const std::size_t y = e / (w - 1);
      const std::size_t x = e % (w - 1);
      v0 = vid2(y, x, w);
      v1 = vid2(y, x + 1, w);
      return;
    }
    const std::size_t ev = e - n_eh;
    const std::size_t y = ev / w;
    const std::size_t x = ev % w;
    v0 = vid2(y, x, w);
    v1 = vid2(y + 1, x, w);
  };

  auto face_boundary_edges = [&](std::size_t f, std::size_t out[4]) {
    const std::size_t y = f / (w - 1);
    const std::size_t x = f % (w - 1);
    out[0] = ehid2(y, x, w);             // top
    out[1] = ehid2(y + 1, x, w);         // bottom
    out[2] = n_eh + y * w + x;           // left vertical
    out[3] = n_eh + y * w + (x + 1);     // right vertical
  };

  for (const auto& alpha : cells) {
    if (alpha.dim == 0) {
      // alpha is vertex: choose best unpaired coface edge incident to alpha.
      const std::size_t v = alpha.id;
      if (g.v2e[v] >= 0) {
        continue;
      }
      const std::size_t y = v / w;
      const std::size_t x = v % w;

      std::size_t best_e = n_e;
      double best_key = -std::numeric_limits<double>::infinity();

      auto try_edge = [&](std::size_t e) {
        // A 1-cell can be matched at most once in the Hasse matching.
        if (e >= n_e || g.e2v[e] >= 0 || g.e2f[e] >= 0) {
          return;
        }
        const double k = (e < n_eh) ? fil.edge_h[e] : fil.edge_v[e - n_eh];
        if (k > best_key || (k == best_key && e > best_e)) {
          best_key = k;
          best_e = e;
        }
      };

      if (x > 0) try_edge(ehid2(y, x - 1, w));
      if (x + 1 < w) try_edge(ehid2(y, x, w));
      if (y > 0) try_edge(n_eh + (y - 1) * w + x);
      if (y + 1 < h) try_edge(n_eh + y * w + x);

      if (best_e < n_e) {
        g.v2e[v] = static_cast<std::int64_t>(best_e);
        g.e2v[best_e] = static_cast<std::int64_t>(v);
      }
      continue;
    }

    if (alpha.dim == 1) {
      // alpha is edge: if unpaired on upper side, pair to best unpaired face coface.
      const std::size_t e = alpha.id;
      if (g.e2f[e] >= 0) {
        continue;
      }

      std::size_t best_f = n_f;
      double best_key = -std::numeric_limits<double>::infinity();

      auto consider_face = [&](std::size_t f) {
        if (f >= n_f || g.f2e[f] >= 0) {
          return;
        }

        // Robins-style unique gradient arrow condition:
        // in face boundary, current edge should be the unique best available edge.
        std::size_t bnd[4] = {0, 0, 0, 0};
        face_boundary_edges(f, bnd);
        std::size_t best_e = n_e;
        double best_e_key = -std::numeric_limits<double>::infinity();
        for (std::size_t ee : bnd) {
          if (g.e2v[ee] >= 0 || g.e2f[ee] >= 0) {
            continue;
          }
          const double ek = (ee < n_eh) ? fil.edge_h[ee] : fil.edge_v[ee - n_eh];
          if (ek > best_e_key || (ek == best_e_key && ee > best_e)) {
            best_e_key = ek;
            best_e = ee;
          }
        }
        if (best_e != e) {
          return;
        }

        const double k = fil.face[f];
        if (k > best_key || (k == best_key && f > best_f)) {
          best_key = k;
          best_f = f;
        }
      };

      if (e < n_eh) {
        // horizontal edge belongs to up to 2 faces: above/below
        const std::size_t y = e / (w - 1);
        const std::size_t x = e % (w - 1);
        if (y > 0) consider_face(fid2(y - 1, x, w));
        if (y + 1 < h) consider_face(fid2(y, x, w));
      } else {
        // vertical edge belongs to up to 2 faces: left/right
        const std::size_t ev = e - n_eh;
        const std::size_t y = ev / w;
        const std::size_t x = ev % w;
        if (x > 0 && y + 1 < h) consider_face(fid2(y, x - 1, w));
        if (x + 1 < w && y + 1 < h) consider_face(fid2(y, x, w));
      }

      if (best_f < n_f) {
        g.e2f[e] = static_cast<std::int64_t>(best_f);
        g.f2e[best_f] = static_cast<std::int64_t>(e);
      }
      continue;
    }

    // alpha is face: no higher coface in 2D cubical complex.
    // Keep as critical if unpaired.
  }

  return g;
}

Filtration3D build_filtration_3d_upper_star(
    const double* prob,
    std::size_t d,
    std::size_t h,
    std::size_t w,
    double tie_eps) {
  Filtration3D out;
  out.d = d;
  out.h = h;
  out.w = w;

  if (prob == nullptr || d == 0 || h == 0 || w == 0) {
    return out;
  }

  const std::size_t n_v = d * h * w;
  const std::size_t n_ex = (w >= 2) ? d * h * (w - 1) : 0;
  const std::size_t n_ey = (h >= 2) ? d * (h - 1) * w : 0;
  const std::size_t n_ez = (d >= 2) ? (d - 1) * h * w : 0;
  const std::size_t n_fxy = (h >= 2 && w >= 2) ? d * (h - 1) * (w - 1) : 0;
  const std::size_t n_fxz = (d >= 2 && w >= 2) ? (d - 1) * h * (w - 1) : 0;
  const std::size_t n_fyz = (d >= 2 && h >= 2) ? (d - 1) * (h - 1) * w : 0;
  const std::size_t n_c = (d >= 2 && h >= 2 && w >= 2) ? (d - 1) * (h - 1) * (w - 1) : 0;

  out.vertex.resize(n_v);
  out.vertex_raw.resize(n_v);
  out.edge_x.resize(n_ex);
  out.edge_y.resize(n_ey);
  out.edge_z.resize(n_ez);
  out.face_xy.resize(n_fxy);
  out.face_xz.resize(n_fxz);
  out.face_yz.resize(n_fyz);
  out.cube.resize(n_c);

  const std::size_t base_v = 0;
  const std::size_t base_ex = base_v + n_v;
  const std::size_t base_ey = base_ex + n_ex;
  const std::size_t base_ez = base_ey + n_ey;
  const std::size_t base_fxy = base_ez + n_ez;
  const std::size_t base_fxz = base_fxy + n_fxy;
  const std::size_t base_fyz = base_fxz + n_fxz;
  const std::size_t base_c = base_fyz + n_fyz;

  for (std::size_t z = 0; z < d; ++z) {
    for (std::size_t y = 0; y < h; ++y) {
      for (std::size_t x = 0; x < w; ++x) {
        const std::size_t i = vid3(z, y, x, h, w);
        const double p = clamp01(prob[i]);
        out.vertex_raw[i] = p;
        out.vertex[i] = perturb(p, base_v + i, tie_eps);
      }
    }
  }

  auto p = [&](std::size_t z, std::size_t y, std::size_t x) {
    return out.vertex_raw[vid3(z, y, x, h, w)];
  };

  for (std::size_t z = 0; z < d; ++z) {
    for (std::size_t y = 0; y < h; ++y) {
      for (std::size_t x = 0; x + 1 < w; ++x) {
        const std::size_t i = (z * h + y) * (w - 1) + x;
        out.edge_x[i] = perturb(std::min(p(z, y, x), p(z, y, x + 1)), base_ex + i, tie_eps);
      }
    }
  }

  for (std::size_t z = 0; z < d; ++z) {
    for (std::size_t y = 0; y + 1 < h; ++y) {
      for (std::size_t x = 0; x < w; ++x) {
        const std::size_t i = (z * (h - 1) + y) * w + x;
        out.edge_y[i] = perturb(std::min(p(z, y, x), p(z, y + 1, x)), base_ey + i, tie_eps);
      }
    }
  }

  for (std::size_t z = 0; z + 1 < d; ++z) {
    for (std::size_t y = 0; y < h; ++y) {
      for (std::size_t x = 0; x < w; ++x) {
        const std::size_t i = (z * h + y) * w + x;
        out.edge_z[i] = perturb(std::min(p(z, y, x), p(z + 1, y, x)), base_ez + i, tie_eps);
      }
    }
  }

  for (std::size_t z = 0; z < d; ++z) {
    for (std::size_t y = 0; y + 1 < h; ++y) {
      for (std::size_t x = 0; x + 1 < w; ++x) {
        const std::size_t i = (z * (h - 1) + y) * (w - 1) + x;
        const double v = std::min(std::min(p(z, y, x), p(z, y, x + 1)), std::min(p(z, y + 1, x), p(z, y + 1, x + 1)));
        out.face_xy[i] = perturb(v, base_fxy + i, tie_eps);
      }
    }
  }

  for (std::size_t z = 0; z + 1 < d; ++z) {
    for (std::size_t y = 0; y < h; ++y) {
      for (std::size_t x = 0; x + 1 < w; ++x) {
        const std::size_t i = (z * h + y) * (w - 1) + x;
        const double v = std::min(std::min(p(z, y, x), p(z, y, x + 1)), std::min(p(z + 1, y, x), p(z + 1, y, x + 1)));
        out.face_xz[i] = perturb(v, base_fxz + i, tie_eps);
      }
    }
  }

  for (std::size_t z = 0; z + 1 < d; ++z) {
    for (std::size_t y = 0; y + 1 < h; ++y) {
      for (std::size_t x = 0; x < w; ++x) {
        const std::size_t i = (z * (h - 1) + y) * w + x;
        const double v = std::min(std::min(p(z, y, x), p(z, y + 1, x)), std::min(p(z + 1, y, x), p(z + 1, y + 1, x)));
        out.face_yz[i] = perturb(v, base_fyz + i, tie_eps);
      }
    }
  }

  for (std::size_t z = 0; z + 1 < d; ++z) {
    for (std::size_t y = 0; y + 1 < h; ++y) {
      for (std::size_t x = 0; x + 1 < w; ++x) {
        const std::size_t i = (z * (h - 1) + y) * (w - 1) + x;
        const double v = std::min(
            std::min(std::min(p(z, y, x), p(z, y, x + 1)), std::min(p(z, y + 1, x), p(z, y + 1, x + 1))),
            std::min(std::min(p(z + 1, y, x), p(z + 1, y, x + 1)), std::min(p(z + 1, y + 1, x), p(z + 1, y + 1, x + 1))));
        out.cube[i] = perturb(v, base_c + i, tie_eps);
      }
    }
  }

  return out;
}

GradientField3D build_gradient_field_3d_scaffold(const Filtration3D& fil) {
  GradientField3D g;

  const std::size_t d = fil.d;
  const std::size_t h = fil.h;
  const std::size_t w = fil.w;

  const std::size_t n_v = d * h * w;
  const std::size_t n_ex = (w >= 2) ? d * h * (w - 1) : 0;
  const std::size_t n_ey = (h >= 2) ? d * (h - 1) * w : 0;
  const std::size_t n_ez = (d >= 2) ? (d - 1) * h * w : 0;
  const std::size_t n_e = n_ex + n_ey + n_ez;

  const std::size_t n_fxy = (h >= 2 && w >= 2) ? d * (h - 1) * (w - 1) : 0;
  const std::size_t n_fxz = (d >= 2 && w >= 2) ? (d - 1) * h * (w - 1) : 0;
  const std::size_t n_fyz = (d >= 2 && h >= 2) ? (d - 1) * (h - 1) * w : 0;
  const std::size_t n_f = n_fxy + n_fxz + n_fyz;

  const std::size_t n_c = (d >= 2 && h >= 2 && w >= 2) ? (d - 1) * (h - 1) * (w - 1) : 0;

  g.v2e.assign(n_v, -1);
  g.e2v.assign(n_e, -1);
  g.e2f.assign(n_e, -1);
  g.f2e.assign(n_f, -1);
  g.f2c.assign(n_f, -1);
  g.c2f.assign(n_c, -1);

  return g;
}

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

  const bool use_h0 = std::find(homology_dims.begin(), homology_dims.end(), 0) != homology_dims.end();
  const bool use_h1 = std::find(homology_dims.begin(), homology_dims.end(), 1) != homology_dims.end();

  const auto fil = build_filtration_2d_upper_star(prob, h, w, 1e-12);
  const auto grad = build_gradient_field_2d_robins(fil);

  out.filtration_values.resize(h * w);
  for (std::size_t i = 0; i < h * w; ++i) {
    out.filtration_values[i] = 1.0 - fil.vertex_raw[i];
  }

  if (!use_h0 && !use_h1) {
    return out;
  }

  // Step 1 (stabilization): deterministic non-empty H0 via union-find sweep on vertices.
  // This gives robust component merge pairs and removes the regression to empty outputs.
  if (use_h0) {
    const std::size_t n_v = h * w;
    DSU dsu(n_v);
    std::vector<std::size_t> order(n_v);
    for (std::size_t i = 0; i < n_v; ++i) {
      order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      return fil.vertex[a] > fil.vertex[b];
    });

    auto union_roots = [&](std::int64_t ra, std::int64_t rb, double death_val) {
      ra = dsu.find(ra);
      rb = dsu.find(rb);
      if (ra == rb) {
        return;
      }

      const double ba = dsu.birth_val[static_cast<std::size_t>(ra)];
      const double bb = dsu.birth_val[static_cast<std::size_t>(rb)];

      std::int64_t keep = ra;
      std::int64_t kill = rb;
      if (bb > ba || (bb == ba && rb < ra)) {
        keep = rb;
        kill = ra;
      }

      const double birth = dsu.birth_val[static_cast<std::size_t>(kill)];
      const double persistence = birth - death_val;
      if (persistence >= min_persistence) {
        out.pairs.push_back(birth);
        out.pairs.push_back(death_val);
        out.dimensions.push_back(0);
        out.birth_indices.push_back(dsu.birth_vertex[static_cast<std::size_t>(kill)]);
        out.death_indices.push_back(-1);
      }

      dsu.parent[static_cast<std::size_t>(kill)] = keep;
      if (dsu.rank[static_cast<std::size_t>(keep)] == dsu.rank[static_cast<std::size_t>(kill)]) {
        dsu.rank[static_cast<std::size_t>(keep)]++;
      }
    };

    for (std::size_t v : order) {
      dsu.activate(static_cast<std::int64_t>(v), fil.vertex[v], static_cast<std::int64_t>(v));
      const std::size_t y = v / w;
      const std::size_t x = v % w;

      // Neighbor unions are triggered at current vertex filtration (upper-star sweep).
      if (x > 0) {
        const std::size_t u = vid2(y, x - 1, w);
        if (dsu.active(static_cast<std::int64_t>(u))) {
          union_roots(static_cast<std::int64_t>(v), static_cast<std::int64_t>(u), fil.vertex[v]);
        }
      }
      if (x + 1 < w) {
        const std::size_t u = vid2(y, x + 1, w);
        if (dsu.active(static_cast<std::int64_t>(u))) {
          union_roots(static_cast<std::int64_t>(v), static_cast<std::int64_t>(u), fil.vertex[v]);
        }
      }
      if (y > 0) {
        const std::size_t u = vid2(y - 1, x, w);
        if (dsu.active(static_cast<std::int64_t>(u))) {
          union_roots(static_cast<std::int64_t>(v), static_cast<std::int64_t>(u), fil.vertex[v]);
        }
      }
      if (y + 1 < h) {
        const std::size_t u = vid2(y + 1, x, w);
        if (dsu.active(static_cast<std::int64_t>(u))) {
          union_roots(static_cast<std::int64_t>(v), static_cast<std::int64_t>(u), fil.vertex[v]);
        }
      }
    }
  }

  // Step 2 (placeholder): derive H1 critical-face candidates from unmatched gradient map.
  // Full Morse boundary/V-path reduction is next milestone; currently we expose deterministic
  // candidate pairs to keep dimension-1 path wired for upcoming strict implementation.
  if (use_h1) {
    const std::size_t n_f = grad.f2e.size();
    for (std::size_t f = 0; f < n_f; ++f) {
      if (grad.f2e[f] >= 0) {
        continue;
      }
      const double birth = (f < fil.face.size()) ? fil.face[f] : 0.0;
      const double death = birth;
      if ((birth - death) < min_persistence) {
        continue;
      }
      out.pairs.push_back(birth);
      out.pairs.push_back(death);
      out.dimensions.push_back(1);
      out.birth_indices.push_back(static_cast<std::int64_t>(f));
      out.death_indices.push_back(-1);
    }
  }

  return out;
}

} // namespace dmt_implicit
