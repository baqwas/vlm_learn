#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <numeric>
#include <omp.h>

namespace py = pybind11;

float compute_mean(py::array_t<float> input_array) {
    auto buf = input_array.request();
    float *ptr = static_cast<float *>(buf.ptr);
    if (buf.size == 0) return 0.0f;
    return std::accumulate(ptr, ptr + buf.size, 0.0f) / buf.size;
}

void normalize_image(py::array_t<float> image, float mean, float std) {
    auto buf = image.request();
    float *ptr = static_cast<float *>(buf.ptr);
    size_t size = buf.size;

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        ptr[i] = ((ptr[i] / 255.0f) - mean) / std;
    }
}

PYBIND11_MODULE(vlm_engine, m) {
    m.doc() = "VLM-Learn Native Engine: Core Numerical Tools";
    m.def("compute_mean", &compute_mean, "Calculates the mean of a NumPy array");
    m.def("normalize_image", &normalize_image, "Normalizes image buffer in-place",
          py::arg("image"), py::arg("mean"), py::arg("std"));
}
