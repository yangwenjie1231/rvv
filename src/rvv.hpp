#pragma once
#include <cstddef>
#include <cstdint>

namespace rvv::core {

/**
 * 向量加法 c = a + b
 * @param a   输入向量 a
 * @param b   输入向量 b
 * @param c   输出向量
 * @param n   元素个数
 * @param module rvv.core.add
 */
void add(const float* a, const float* b, float* c, std::size_t n);

/**
 * 向量减法 c = a - b
 * @module rvv.core.sub
 */
void sub(const float* a, const float* b, float* c, std::size_t n);

/**
 * 标量乘法 b = k * a
 * @module rvv.core.scale
 */
void scale(const float* a, float k, float* b, std::size_t n);

/**
 * 向量点积
 * @return 点积结果
 * @module rvv.core.dot
 */
float dot(const float* a, const float* b, std::size_t n);

/**
 * L2 范数 ||a||_2
 * @module rvv.core.norm_l2
 */
float norm_l2(const float* a, std::size_t n);

/**
 * 向量归一化 b = a / ||a||_2
 * @module rvv.core.normalize
 */
void normalize(const float* a, float* b, std::size_t n);

/**
 * 矩阵加法 C = A + B
 * @param rows 行数
 * @param cols 列数
 * @module rvv.core.add2d
 */
void add2d(const float* A, const float* B, float* C,
           std::size_t rows, std::size_t cols);

/**
 * 矩阵标量乘法 B = k * A
 * @module rvv.core.scale2d
 */
void scale2d(const float* A, float k, float* B,
             std::size_t rows, std::size_t cols);

/**
 * 矩阵乘法 C = A * B
 * A:[rows×k]  B:[k×cols]  → C:[rows×cols]
 * @module rvv.core.matmul
 */
void matmul(const float* A, const float* B, float* C,
            std::size_t rows, std::size_t k, std::size_t cols);

/**
 * 矩阵转置 B = A^T
 * @module rvv.core.transpose
 */
void transpose(const float* A, float* B,
               std::size_t rows, std::size_t cols);

/**
 * 矩阵 × 向量  y = A * x
 * A:[rows×cols]  x:[cols]  → y:[rows]
 * @module rvv.core.mv
 */
void mv(const float* A, const float* x, float* y,
        std::size_t rows, std::size_t cols);

}  // namespace rvv::core