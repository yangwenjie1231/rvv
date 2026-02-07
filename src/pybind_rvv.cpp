#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include "rvv.hpp"

namespace py = pybind11;

// 工具：把任意 float32 ndarray 转为一维连续 vector
std::vector<float> np_to_vec(py::array_t<float> a) {
    py::buffer_info info = a.request();
    if (info.ndim != 1)
        throw std::runtime_error("Only 1-D array accepted");
    float* ptr = static_cast<float*>(info.ptr);
    return std::vector<float>(ptr, ptr + info.size);
}

// 工具：把结果写回 numpy
py::array_t<float> vec_to_np(const std::vector<float>& v) {
    py::array_t<float> out(v.size());
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

//--------------------------------------
// 向量运算封装
//--------------------------------------
py::array_t<float> py_add(py::array_t<float> a, py::array_t<float> b) {
    auto va = np_to_vec(a);
    auto vb = np_to_vec(b);
    if (va.size() != vb.size())
        throw std::runtime_error("Shape mismatch");
    std::vector<float> vc(va.size());
    rvv::core::add(va.data(), vb.data(), vc.data(), va.size());
    return vec_to_np(vcire);
}

py::array_t<float> py_sub(py::array_t<float> a, py::array_t<float> b) {
    auto va = np_to_vec(a);
    auto vb = np_to_vec(b);
    if (va.size() != vb.size())
        throw std::runtime_error("Shape mismatch");
    std::vector<float> vc(va.size());
    rvv::core::sub(va.data(), vb.data(), vc.data(), va.size());
    return vec_to_np(vc);
}

py::array_t<float> py_scale(py::array_t<float> a, float k) {
    auto va = np_to_vec(a);
    std::vector<float> vb(va.size());
    rvv::core::scale(va.data(), k, vb.data(), va.size());
    return vec_to_np(vb);
}

float py_dot(py::array_t<float> a, py::array_t<float> b) {
    auto va = np_to_vec(a);
    auto vb = np_to_vec(b);
    if (va.size() != vb.size())
        throw std::runtime_error("Shape mismatch");
    return rvv::core::dot(va.data(), vb.data(), va.size());
}

float py_norm_l2(py::array_t<float> a) {
    auto va = np_to_vec(a);
    return rvv::core::norm_l2(va.data(), va.size());
}

py::array_t<float> py_normalize(py::array_t<float> a) {
    auto va = np_to_vec(a);
    std::vector<float> vb(va.size());
    rvv::core::normalize(va.data(), vb.data(), va.size());
    return vec_to_np(vb);
}

//--------------------------------------
// 矩阵运算封装（2-D array）
//--------------------------------------
using RowMajor = py::array::f_style;  // numpy 默认 C-style，这里显式指明行主序
using MatF   = py::array_t<float, py::array::c_style | py::array::forcecast>;

std::vector<float> mat_to_vec(const MatF& m) {
    py::buffer_info info = m.request();
    if (info.ndim != 2)
        throw std::runtime_error("Need 2-D array");
    return std::vector<float>(static_cast<float*>(info.ptr),
                              static_cast<float*>(info.ptr) + info.size);
}

MatF vec_to_mat(const std::vector<float>& v, std::size_t rows, std::size_t cols) {
    MatF out({rows, cols});
    std::copy(v.begin(), v.end(), out.mutable_data());
    return out;
}

MatF py_add2d(MatF A, MatF B) {
    auto vA = mat_to_vec(A);
    auto vB = mat_to_vec(B);
    if (vA.size() != vB.size())
        throw std::runtime_error("Shape mismatch");
    std::vector<float> vC(vA.size());
    py::buffer_info info = A.request();
    rvv::core::add2d(vA.data(), vB.data(), vC.data(),
                     info.shape[0], info.shape[1]);
    return vec_to_mat(vC, info.shape[0], info.shape[1]);
}

MatF py_scale2d(MatF A, float k) {
    auto vA = mat_to_vec(A);
    std::vector<float> vB(vA.size());
    py::buffer_info info = A.request();
    rvv::core::scale2d(vA.data(), k, vB.data(),
                       info.shape[0], info.shape[1]);
    return vec_to_mat(vB, info.shape[0], info.shape[1]);
}

MatF py_matmul(MatF A, MatF B) {
    py::buffer_info infoA = A.request();
    py::buffer_info infoB = B.request();
    if (infoA.ndim != 2 || infoB.ndim != 2)
        throw std::runtime_error("Need 2-D arrays");
    std::size_t rows = infoA.shape[0];
    std::size_t k    = infoA.shape[1];
    std::size_t cols = infoB.shape[1];
    if (infoB.shape[0] != k)
        throw std::runtime_error("Incompatible shapes for matmul");
    auto vA = mat_to_vec(A);
    auto vB = mat_to_vec(B);
    std::vector<float> vC(rows * cols);
    rvv::core::matmul(vA.data(), vB.data(), vC.data(), rows, k, cols);
    return vec_to_mat(vC, rows, cols);
}

MatF py_transpose(MatF A) {
    py::buffer_info info = A.request();
    std::size_t rows = info.shape[0];
    std::size_t cols = info.shape[1];
    auto vA = mat_to_vec(A);
    std::vector<float> vB(rows * cols);
    rvv::core::transpose(vA.data(), vB.data(), rows, cols);
    return vec_to_mat(vB, cols, rows);
}

py::array_t<float> py_mv(MatF A, py::array_t<float> x) {
    py::buffer_info infoA = A.request();
    auto vx = np_to_vec(x);
    if (infoA.ndim != 2 || vx.size() != infoA.shape[1])
        throw std::runtime_error("Shape mismatch");
    std::size_t rows = infoA.shape[0];
    std::size_t cols = infoA.shape[1];
    auto vA = mat_to_vec(A);
    std::vector<float> vy(rows);
    rvv::core::mv(vA.data(), vx.data(), vy.data(), rows, cols);
    return vec_to_np(vy);
}

//--------------------------------------
// Python 模块定义
//--------------------------------------
PYBIND11_MODULE(rvv, m) {
    m.doc() = "SG2002 RVV 0.7.1 加速库，兼容 NumPy";

    // 向量
    m.def("add",      &py_add,      "向量加法");
    m.def("sub",      &py_sub,      "向量减法");
    m.def("scale",    &py_scale,    "标量乘法");
    m.def("dot",      &py_dot,      "点积");
    m.def("norm_l2",  &py_norm_l2,  "L2 范数");
    m.def("normalize",&py_normalize,"向量归一化");

    // 矩阵
    m.def("add2d",    &py_add2d,    "矩阵加法");
    m.def("scale2d",  &py_scale2d,  "矩阵标量乘法");
    m.def("matmul",   &py_matmul,   "矩阵乘法");
    m.def("transpose",&py_transpose,"矩阵转置");
    m.def("mv",       &py_mv,       "矩阵 × 向量");
}