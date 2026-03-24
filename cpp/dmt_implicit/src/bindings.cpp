#include "dmt_implicit.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace {

py::dict extract_persistence_2d_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> prob,
    double min_persistence,
    std::vector<int> homology_dims) {
  if (prob.ndim() != 2) {
    throw std::invalid_argument("extract_persistence_2d expects a 2D array [H, W]");
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

  const auto out = dmt_implicit::extract_persistence_2d(
      flat.data(), h, w, min_persistence, homology_dims);

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

} // namespace

PYBIND11_MODULE(dmt_implicit_ext, m) {
  m.doc() = "Prototype implicit DMT backend extension";
  m.def(
      "extract_persistence_2d",
      &extract_persistence_2d_py,
      py::arg("prob"),
      py::arg("min_persistence") = 0.0,
      py::arg("homology_dims") = std::vector<int>{0, 1});
}
