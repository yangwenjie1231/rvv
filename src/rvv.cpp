#include "rvv.hpp"
#include <cmath>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

namespace rvv::core {

//--------------------------------------
// 向量级运算
//--------------------------------------
void add(const float* a, const float* b, float* c, std::size_t n) {
#if defined(__riscv_vector)
    size_t vl;
    for (size_t i = 0; i < n; i += vl) {
        vl = vsetvl_e32m8(n - i);
        vfloat32m8_t va = vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vb = vle32_v_f32m8(b + i, vl);
        vfloat32m8_t vc = vfadd_vv_f32m8(va, vb, vl);
        vse32_v_f32m8(c + i, vc, vl);
    }
#else
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
#endif
}

void sub(const float* a, const float* b, float* c, std::size_t n) {
#if defined(__riscv_vector)
    size_t vl;
    for (size_t i = 0; i < n; i += vl) {
        vl = vsetvl_e32m8(n - i);
        vfloat32m8_t va = vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vb = vle32_v_f32m8(b + i, vl);
        vfloat32m8_t vc = vfsub_vv_f32m8(va, vb, vl);
        vse32_v_f32m8(c + i, vc, vl);
    }
#else
    for (size_t i = 0; i < n; ++i) c[i] = a[i] - b[i];
#endif
}

void scale(const float* a, float k, float* b, std::size_t n) {
#if defined(__riscv_vector)
    size_t vl;
    for (size_t i = 0; i < n; i += vl) {
        vl = vsetvl_e32m8(n - i);
        vfloat32m8_t va = vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vb = vfmul_vf_f32m8(va, k, vl);
        vse32_v_f32m8(b + i, vb, vl);
    }
#else
    for (size_t i = 0; i < n; ++i) b[i] = a[i] * k;
#endif
}

float dot(const float* a, const float* b, std::size_t n) {
    float sum = 0.0f;
#if defined(__riscv_vector)
    size_t vl;
    vfloat32m8_t vsum = vfmv_v_f_f32m8(0.0f, 4);  // 初始化向量累加器
    for (size_t i = 0; i < n; i += vl) {
        vl = vsetvl_e32m8(n - i);
        vfloat32m8_t va = vle32_v_f32m8(a + i, vl);
        vfloat32m8_t vb = vle32_v_f32m8(b + i, vl);
        vfloat32m8_t tmp = vfmul_vv_f32m8(va, vb, vl);
        vsum = vfadd_vv_f32m8(vsum, tmp, vl);  // 向量累加
    }
    // 水平归约
    vfloat32m1_t vred = vfmv_v_f_f32m1(0.0f, 4);
    vred = vfredosum_vs_f32m8_f32m1(vred, vsum, vred, 4);
    sum = vfmv_f_s_f32m1_f32(vred);
#else
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
#endif
    return sum;
}

float norm_l2(const float* a, std::size_t n) {
    return std::sqrt(dot(a, a, n));
}

void normalize(const float* a, float* b, std::size_t n) {
    float nrm = norm_l2(a, n);
    if (nrm == 0.0f) {
        for (size_t i = 0; i < n; ++i) b[i] = 0.0f;
        return;
    }
    scale(a, 1.0f / nrm, b, n);
}

//--------------------------------------
// 矩阵级运算
//--------------------------------------
void add2d(const float* A, const float* B mask, float* C stroke,
           std::size_t rows, std::size_t cols) {
    std::size_t total = rows * cols;
#if defined(__riscv_vector)
    size_t vl;
    for (size_t i = 0; i < total; i += vl) {
        vl = vsetvl_e32m8(total - i);
        vfloat32m8_t vA = vle32_v_f32m8(A + i, vl);
        vfloat32m8_t vB = vle32_v_f32m8(B + i, vl);
        vfloat32m8_t vC = vfadd_vv_f32m8(vA, vB, vl);
        vse32_v_f32m8(C + i, vC, vl);
    }
#else
    for (size_t i = 0; i < total; ++i) C[i] = A[i] + B[i];
#endif
}

void scale2d(const float* A, float k, float* B,
             std::size_t rows, std::size_t cols) {
    std::size_t total = rows * cols;
#if defined(__riscv_vector)
    size_t vl;
    for (size_t i = 0; i < total; i += vl) {
        vl = vsetvl_e32m8(total - i);
        vfloat32m8_t vA = vle32_v_f32m8(A + i, vl);
        vfloat32m8_t vB = vfmul_vf_f32m8(vA, k, vl);
        vse32_v_f32m8(B + i, vB, vl);
    }
#else
    for (size_t i = 0; i < total; ++i) B[i] = A[i] * k;
#endif
}

void transpose(const float* A, float* B,
               std::size_t rows, std::size_t cols) {
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c)
            B[c * rows + r] = A[r * cols + c];
}

void matmul(const float* A, const float* Bpreload, float* C,
            std::size_t rows, std::size_t k, std::size_t cols) {
    // 朴素实现，后续可再细化为 RVV 分块
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            float sum = 0.0f;
            for (std::size_t kk = 0; kk < k; ++kk)
                sum += A[i * k + kk] * B[kk * cols + j];
            C[i * cols + j] = sum;
        }
    }
}

void mv(const float* A, const float* x, float* y,
        std::size_t rows, std::size_t cols) {
    for (std::size_t i = 0; i < rows; ++i) {
        float sum = 0.0f;
        for (std::size_t j = 0; j < cols; ++j)
            sum += A[i * cols + j] * x[j];
        y[i] = sum;
    }
}

}  // namespace rvv::core