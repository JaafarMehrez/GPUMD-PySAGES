/*
    gpumd_python.cpp

    pybind11 wrapper for GPUMD that exposes simulation data as DLPack
    capsules and supports a per-step Python callback.

    Compile this file with a C++ compiler (g++/clang++) and link it with
    the GPUMD object files to produce a loadable Python module named
    ``gpumd``.

    Copyright 2026 Jaafar Mehrez
    (Shanghai Jiao Tong University, Shanghai, China;
     HPQC Labs, Waterloo, Canada;
     jaafarmehrez@sjtu.edu.cn, jaafar@hpqc.org)

    SPDX-License-Identifier: MIT
*/

#include <cuda_runtime.h>

#include <cctype>   // std::tolower
#include <cstring>  // std::memset
#include <fstream>  // std::ifstream
#include <memory>   // std::unique_ptr, std::make_unique

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "dlpack.h"
#include "gpumd_python_kernels.cuh"
#include "run.cuh"
#include "utilities/gpu_vector.cuh"

namespace py = pybind11;

// Helper: wrap a GPU_Vector as a DLPack capsule.
template <typename T>
py::capsule gpu_vector_to_dlpack(
  T* data,
  int64_t size,
  int64_t components,                     // 3 for per-atom vectors, 1 for scalars
  DLDataTypeCode type_code,
  uint8_t bits)
{
  int ndim = (components > 1) ? 2 : 1;
  int64_t* shape = new int64_t[ndim];
  int64_t* strides = new int64_t[ndim];
  if (ndim == 2) {
    shape[0] = components;   // 3
    shape[1] = size;         // N
    strides[0] = size;       // N
    strides[1] = 1;          // 1
  } else {
    shape[0] = size;
    strides[0] = 1;
  }

  auto* manager = new DLManagedTensor;
  std::memset(manager, 0, sizeof(DLManagedTensor));
  manager->dl_tensor.data = static_cast<void*>(data);
  manager->dl_tensor.device = DLDevice{kDLCUDA, 0};
  manager->dl_tensor.ndim = ndim;
  manager->dl_tensor.dtype.code = static_cast<uint8_t>(type_code);
  manager->dl_tensor.dtype.bits = bits;
  manager->dl_tensor.dtype.lanes = 1;
  manager->dl_tensor.shape = shape;
  manager->dl_tensor.strides = strides;
  manager->dl_tensor.byte_offset = 0;
  manager->manager_ctx = nullptr;
  manager->deleter = [](DLManagedTensor* self) {
    delete[] self->dl_tensor.shape;
    delete[] self->dl_tensor.strides;
    delete self;
  };
  py::capsule cap(manager, "dltensor", [](PyObject* /*capsule*/) {
  });
  return cap;
}

// Convenience overloads for the common GPUMD types.
py::capsule dlpack_from_double_vector(double* data, int64_t n, int64_t comp = 1)
{
  return gpu_vector_to_dlpack(data, n, comp, kDLFloat, 64);
}

py::capsule dlpack_from_int_vector(int* data, int64_t n, int64_t comp = 1)
{
  return gpu_vector_to_dlpack(data, n, comp, kDLInt, 32);
}

// PySimulation: high-level wrapper exposed to Python.
class PySimulation
{
  std::unique_ptr<Run> run_;
  std::string run_input_path_;

public:
  explicit PySimulation(const std::string& run_input_path = "run.in")
    : run_input_path_(run_input_path)
  {
    run_ = std::make_unique<Run>(true, run_input_path);
    cudaDeviceSynchronize();
  }

  // Query whether the simulation box is guaranteed to remain constant.
  bool is_box_constant() const
  {
    std::ifstream file(run_input_path_);
    if (!file.is_open()) {
      return false;
    }
    std::string line;
    while (std::getline(file, line)) {
      auto first_non_space = line.find_first_not_of(" \t\r\n");
      if (first_non_space == std::string::npos) continue;
      if (line[first_non_space] == '#') continue;

      std::string lower;
      for (char c : line) lower += std::tolower(c);

      if (lower.find("ensemble npt") != std::string::npos) return false;
      if (lower.find("change_box") != std::string::npos) return false;
      if (lower.find("puff") != std::string::npos) return false;
    }
    return true;
  }

  // DLPack accessors (zero-copy views into GPUMD GPU memory)
  py::capsule get_positions_dlpack()
  {
    return dlpack_from_double_vector(
      run_->get_atom().position_per_atom.data(),
      run_->get_atom().number_of_atoms,
      3);
  }

  py::capsule get_velocities_dlpack()
  {
    return dlpack_from_double_vector(
      run_->get_atom().velocity_per_atom.data(),
      run_->get_atom().number_of_atoms,
      3);
  }

  py::capsule get_forces_dlpack()
  {
    return dlpack_from_double_vector(
      run_->get_atom().force_per_atom.data(),
      run_->get_atom().number_of_atoms,
      3);
  }

  py::capsule get_masses_dlpack()
  {
    return dlpack_from_double_vector(
      run_->get_atom().mass.data(),
      run_->get_atom().number_of_atoms,
      1);
  }

  py::capsule get_types_dlpack()
  {
    return dlpack_from_int_vector(
      run_->get_atom().type.data(),
      run_->get_atom().number_of_atoms,
      1);
  }

  // Box & timestep
  py::tuple get_box()
  {
    py::list h(9);
    for (int i = 0; i < 9; ++i) {
      h[i] = run_->get_box().cpu_h[i];
    }
    py::tuple origin = py::make_tuple(0.0, 0.0, 0.0);
    return py::make_tuple(h, origin);
  }

  double get_timestep() const
  {
    return run_->get_time_step();
  }

  int get_number_of_atoms() const
  {
    return run_->get_atom().number_of_atoms;
  }

  // Callback registration
  void set_step_callback(std::function<void(int)> cb)
  {
    run_->step_callback = cb;
  }

  // Execution
  void run(int steps)
  {
    run_->set_number_of_steps(steps);
    run_->execute_run();
  }

  void synchronize()
  {
    cudaDeviceSynchronize();
  }

  // Direct bias write (optional)
  void set_external_bias(py::capsule dlpack_cap)
  {
    DLManagedTensor* dlm = static_cast<DLManagedTensor*>(dlpack_cap.get_pointer());
    if (!dlm) {
      throw std::runtime_error("Invalid DLPack capsule");
    }

    const DLTensor& dt = dlm->dl_tensor;
    if (dt.device.device_type != kDLCUDA && dt.device.device_type != kDLCUDAManaged) {
      throw std::runtime_error("set_external_bias expects a CUDA DLPack tensor");
    }

    int64_t expected = run_->get_atom().number_of_atoms * 3;
    int64_t actual = 1;
    for (int i = 0; i < dt.ndim; ++i) {
      actual *= dt.shape[i];
    }
    if (actual != expected) {
      throw std::runtime_error("Bias tensor size mismatch");
    }

    if (dt.dtype.code != static_cast<uint8_t>(kDLFloat) || dt.dtype.bits != 64) {
      char msg[128];
      snprintf(
        msg,
        sizeof(msg),
        "set_external_bias: expected float64 (kDLFloat,64) bias, got code=%u bits=%u",
        dt.dtype.code,
        dt.dtype.bits);
      throw std::runtime_error(msg);
    }

    cudaMemcpy(
      run_->external_bias_per_atom.data(),
      dt.data,
      expected * sizeof(double),
      cudaMemcpyDeviceToDevice);
  }

  // Zero the external bias buffer
  void clear_external_bias()
  {
    int64_t n = run_->external_bias_per_atom.size();
    if (n > 0) {
      cudaMemset(run_->external_bias_per_atom.data(), 0, n * sizeof(double));
    }
  }

  // Direct in-place bias add via custom CUDA kernel
  void add_aos_bias_to_forces(py::capsule dlpack_cap)
  {
    DLManagedTensor* dlm = static_cast<DLManagedTensor*>(dlpack_cap.get_pointer());
    if (!dlm) {
      throw std::runtime_error("Invalid DLPack capsule");
    }

    const DLTensor& dt = dlm->dl_tensor;
    if (dt.device.device_type != kDLCUDA && dt.device.device_type != kDLCUDAManaged) {
      throw std::runtime_error("add_aos_bias_to_forces expects a CUDA DLPack tensor");
    }

    int64_t expected = run_->get_atom().number_of_atoms * 3;
    int64_t actual = 1;
    for (int i = 0; i < dt.ndim; ++i) {
      actual *= dt.shape[i];
    }
    if (actual != expected) {
      throw std::runtime_error("Bias tensor size mismatch");
    }

    if (dt.dtype.code != static_cast<uint8_t>(kDLFloat) || dt.dtype.bits != 64) {
      char msg[128];
      snprintf(
        msg,
        sizeof(msg),
        "add_aos_bias_to_forces: expected float64 bias, got code=%u bits=%u",
        dt.dtype.code,
        dt.dtype.bits);
      throw std::runtime_error(msg);
    }

    const int N = run_->get_atom().number_of_atoms;
    const double* bias_aos = static_cast<const double*>(dt.data);
    double* forces_soa = run_->get_atom().force_per_atom.data();

    gpu_add_aos_bias_to_soa_forces(N, forces_soa, bias_aos);

    // Synchronize so GPUMD's subsequent integrate step sees the updated forces.
    cudaDeviceSynchronize();
  }
};

// Module definition
PYBIND11_MODULE(gpumd, m)
{
  m.doc() = "GPUMD Python wrapper for PySAGES integration";

  py::class_<PySimulation>(m, "Simulation")
    .def(py::init<const std::string&>(), py::arg("run_input_path") = "run.in")
    .def("get_positions_dlpack", &PySimulation::get_positions_dlpack)
    .def("get_velocities_dlpack", &PySimulation::get_velocities_dlpack)
    .def("get_forces_dlpack", &PySimulation::get_forces_dlpack)
    .def("get_masses_dlpack", &PySimulation::get_masses_dlpack)
    .def("get_types_dlpack", &PySimulation::get_types_dlpack)
    .def("get_box", &PySimulation::get_box)
    .def("get_timestep", &PySimulation::get_timestep)
    .def("get_number_of_atoms", &PySimulation::get_number_of_atoms)
    .def("set_step_callback", &PySimulation::set_step_callback)
    .def("run", &PySimulation::run)
    .def("synchronize", &PySimulation::synchronize)
    .def("set_external_bias", &PySimulation::set_external_bias)
    .def("clear_external_bias", &PySimulation::clear_external_bias)
    .def("add_aos_bias_to_forces", &PySimulation::add_aos_bias_to_forces)
    .def("is_box_constant", &PySimulation::is_box_constant,
         "Returns True if the simulation box is guaranteed to remain constant (NVT/NVE).");
}
