#include "dmt_implicit.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

std::vector<double> flatten_2d(py::array_t<double, py::array::c_style | py::array::forcecast> prob) {
  if (prob.ndim() != 2) {
    throw std::invalid_argument("Expected 2D array [H, W]");
  }
  const auto h = static_cast<std::size_t>(prob.shape(0));
  const auto w = static_cast<std::size_t>(prob.shape(1));
  auto buf = prob.unchecked<2>();
  std::vector<double> flat(h * w);
  for (std::size_t y = 0; y < h; ++y) {
    for (std::size_t x = 0; x < w; ++x) {
      flat[y * w + x] = buf(y, x);
    }
  }
  return flat;
}

std::vector<double> flatten_3d(py::array_t<double, py::array::c_style | py::array::forcecast> vol) {
  if (vol.ndim() != 3) {
    throw std::invalid_argument("Expected 3D array [D, H, W]");
  }
  const auto d = static_cast<std::size_t>(vol.shape(0));
  const auto h = static_cast<std::size_t>(vol.shape(1));
  const auto w = static_cast<std::size_t>(vol.shape(2));
  auto buf = vol.unchecked<3>();
  std::vector<double> flat(d * h * w);
  for (std::size_t z = 0; z < d; ++z) {
    for (std::size_t y = 0; y < h; ++y) {
      for (std::size_t x = 0; x < w; ++x) {
        flat[(z * h + y) * w + x] = buf(z, y, x);
      }
    }
  }
  return flat;
}

py::dict extract_persistence_2d_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> prob,
    double min_persistence,
    std::vector<int> homology_dims) {
  if (prob.ndim() != 2) {
    throw std::invalid_argument("extract_persistence_2d expects a 2D array [H, W]");
  }

  const auto h = static_cast<std::size_t>(prob.shape(0));
  const auto w = static_cast<std::size_t>(prob.shape(1));
  auto flat = flatten_2d(prob);

  const auto out = dmt_implicit::extract_persistence_2d(flat.data(), h, w, min_persistence, homology_dims);

  const std::size_t m = out.dimensions.size();

  py::array_t<double> pairs({static_cast<py::ssize_t>(m), static_cast<py::ssize_t>(2)});
  {
    auto pairs_mut = pairs.mutable_unchecked<2>();
    for (std::size_t i = 0; i < m; ++i) {
      const std::size_t base = i * 2;
      pairs_mut(i, 0) = (base < out.pairs.size()) ? out.pairs[base] : 0.0;
      pairs_mut(i, 1) = (base + 1 < out.pairs.size()) ? out.pairs[base + 1] : 0.0;
    }
  }

  py::array_t<std::int64_t> dimensions({static_cast<py::ssize_t>(m)});
  py::array_t<std::int64_t> birth_indices({static_cast<py::ssize_t>(m)});
  py::array_t<std::int64_t> death_indices({static_cast<py::ssize_t>(m)});
  {
    auto dim_mut = dimensions.mutable_unchecked<1>();
    auto b_mut = birth_indices.mutable_unchecked<1>();
    auto d_mut = death_indices.mutable_unchecked<1>();
    for (std::size_t i = 0; i < m; ++i) {
      dim_mut(i) = out.dimensions[i];
      b_mut(i) = (i < out.birth_indices.size()) ? out.birth_indices[i] : -1;
      d_mut(i) = (i < out.death_indices.size()) ? out.death_indices[i] : -1;
    }
  }

  py::array_t<double> filtration_values({static_cast<py::ssize_t>(h), static_cast<py::ssize_t>(w)});
  {
    auto f_mut = filtration_values.mutable_unchecked<2>();
    for (std::size_t y = 0; y < h; ++y) {
      for (std::size_t x = 0; x < w; ++x) {
        const std::size_t idx = y * w + x;
        f_mut(y, x) = (idx < out.filtration_values.size()) ? out.filtration_values[idx] : 0.0;
      }
    }
  }

  py::dict ret;
  ret["pairs"] = pairs;
  ret["dimensions"] = dimensions;
  ret["birth_indices"] = birth_indices;
  ret["death_indices"] = death_indices;
  ret["filtration_values"] = filtration_values;
  return ret;
}

py::dict debug_build_filtration_2d_py(py::array_t<double, py::array::c_style | py::array::forcecast> prob, double tie_eps) {
  const auto h = static_cast<std::size_t>(prob.shape(0));
  const auto w = static_cast<std::size_t>(prob.shape(1));
  auto flat = flatten_2d(prob);

  const auto fil = dmt_implicit::build_filtration_2d_upper_star(flat.data(), h, w, tie_eps);

  py::array_t<double> vertex({static_cast<py::ssize_t>(fil.vertex.size())});
  py::array_t<double> edge_h({static_cast<py::ssize_t>(fil.edge_h.size())});
  py::array_t<double> edge_v({static_cast<py::ssize_t>(fil.edge_v.size())});
  py::array_t<double> face({static_cast<py::ssize_t>(fil.face.size())});

  auto v = vertex.mutable_unchecked<1>();
  for (std::size_t i = 0; i < fil.vertex.size(); ++i) v(i) = fil.vertex[i];
  auto eh = edge_h.mutable_unchecked<1>();
  for (std::size_t i = 0; i < fil.edge_h.size(); ++i) eh(i) = fil.edge_h[i];
  auto ev = edge_v.mutable_unchecked<1>();
  for (std::size_t i = 0; i < fil.edge_v.size(); ++i) ev(i) = fil.edge_v[i];
  auto f = face.mutable_unchecked<1>();
  for (std::size_t i = 0; i < fil.face.size(); ++i) f(i) = fil.face[i];

  py::dict ret;
  ret["h"] = py::int_(fil.h);
  ret["w"] = py::int_(fil.w);
  ret["vertex"] = vertex;
  ret["edge_h"] = edge_h;
  ret["edge_v"] = edge_v;
  ret["face"] = face;
  return ret;
}

py::dict debug_build_gradient_2d_py(py::array_t<double, py::array::c_style | py::array::forcecast> prob, double tie_eps) {
  const auto h = static_cast<std::size_t>(prob.shape(0));
  const auto w = static_cast<std::size_t>(prob.shape(1));
  auto flat = flatten_2d(prob);

  const auto fil = dmt_implicit::build_filtration_2d_upper_star(flat.data(), h, w, tie_eps);
  const auto g = dmt_implicit::build_gradient_field_2d_robins(fil);

  py::array_t<std::int64_t> v2e({static_cast<py::ssize_t>(g.v2e.size())});
  py::array_t<std::int64_t> e2v({static_cast<py::ssize_t>(g.e2v.size())});
  py::array_t<std::int64_t> e2f({static_cast<py::ssize_t>(g.e2f.size())});
  py::array_t<std::int64_t> f2e({static_cast<py::ssize_t>(g.f2e.size())});

  auto a1 = v2e.mutable_unchecked<1>();
  for (std::size_t i = 0; i < g.v2e.size(); ++i) a1(i) = g.v2e[i];
  auto a2 = e2v.mutable_unchecked<1>();
  for (std::size_t i = 0; i < g.e2v.size(); ++i) a2(i) = g.e2v[i];
  auto a3 = e2f.mutable_unchecked<1>();
  for (std::size_t i = 0; i < g.e2f.size(); ++i) a3(i) = g.e2f[i];
  auto a4 = f2e.mutable_unchecked<1>();
  for (std::size_t i = 0; i < g.f2e.size(); ++i) a4(i) = g.f2e[i];

  py::dict ret;
  ret["v2e"] = v2e;
  ret["e2v"] = e2v;
  ret["e2f"] = e2f;
  ret["f2e"] = f2e;
  return ret;
}

py::dict debug_build_filtration_3d_py(py::array_t<double, py::array::c_style | py::array::forcecast> vol, double tie_eps) {
  const auto d = static_cast<std::size_t>(vol.shape(0));
  const auto h = static_cast<std::size_t>(vol.shape(1));
  const auto w = static_cast<std::size_t>(vol.shape(2));
  auto flat = flatten_3d(vol);

  const auto fil = dmt_implicit::build_filtration_3d_upper_star(flat.data(), d, h, w, tie_eps);

  py::dict ret;
  ret["d"] = py::int_(fil.d);
  ret["h"] = py::int_(fil.h);
  ret["w"] = py::int_(fil.w);
  ret["num_vertex"] = py::int_(fil.vertex.size());
  ret["num_edge_x"] = py::int_(fil.edge_x.size());
  ret["num_edge_y"] = py::int_(fil.edge_y.size());
  ret["num_edge_z"] = py::int_(fil.edge_z.size());
  ret["num_face_xy"] = py::int_(fil.face_xy.size());
  ret["num_face_xz"] = py::int_(fil.face_xz.size());
  ret["num_face_yz"] = py::int_(fil.face_yz.size());
  ret["num_cube"] = py::int_(fil.cube.size());
  return ret;
}

py::dict debug_build_gradient_3d_py(py::array_t<double, py::array::c_style | py::array::forcecast> vol, double tie_eps) {
  const auto d = static_cast<std::size_t>(vol.shape(0));
  const auto h = static_cast<std::size_t>(vol.shape(1));
  const auto w = static_cast<std::size_t>(vol.shape(2));
  auto flat = flatten_3d(vol);

  const auto fil = dmt_implicit::build_filtration_3d_upper_star(flat.data(), d, h, w, tie_eps);
  const auto g = dmt_implicit::build_gradient_field_3d_scaffold(fil);

  py::dict ret;
  ret["v2e_size"] = py::int_(g.v2e.size());
  ret["e2v_size"] = py::int_(g.e2v.size());
  ret["e2f_size"] = py::int_(g.e2f.size());
  ret["f2e_size"] = py::int_(g.f2e.size());
  ret["f2c_size"] = py::int_(g.f2c.size());
  ret["c2f_size"] = py::int_(g.c2f.size());
  return ret;
}

} // namespace

PYBIND11_MODULE(dmt_implicit_ext, m) {
  m.doc() = "Prototype implicit DMT backend extension";
  m.def("extract_persistence_2d", &extract_persistence_2d_py, py::arg("prob"), py::arg("min_persistence") = 0.0, py::arg("homology_dims") = std::vector<int>{0, 1});
  m.def("debug_build_filtration_2d", &debug_build_filtration_2d_py, py::arg("prob"), py::arg("tie_eps") = 1e-12);
  m.def("debug_build_gradient_2d", &debug_build_gradient_2d_py, py::arg("prob"), py::arg("tie_eps") = 1e-12);
  m.def("debug_build_filtration_3d", &debug_build_filtration_3d_py, py::arg("vol"), py::arg("tie_eps") = 1e-12);
  m.def("debug_build_gradient_3d", &debug_build_gradient_3d_py, py::arg("vol"), py::arg("tie_eps") = 1e-12);
}
