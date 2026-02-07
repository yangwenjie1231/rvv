# rvv API 参考

所有函数均支持 `numpy.ndarray(dtype=float32)` 输入/输出。

## 向量运算
- `rvv.add(a, b)` → ndarray  
- `rvv.sub(a, b)` → ndarray  
- `rvv.scale(a, k)` → ndarray  
- `rvv.dot(a, b)` → float  
- `rvv.norm_l2(a)` → float  
- `rvv.normalize(a)` → ndarray  

## 矩阵运算
- `rvv.add2d(A, B)` → ndarray  
- `rvv.scale2d(A, k)` → ndarray  
- `rvv.matmul(A, B)` → ndarray  
- `rvv.transpose(A)` → ndarray  
- `rvv.mv(A, x)` → ndarray  （矩阵 × 向量）

## 示例
```python
import numpy as np, rvv
a = np.ones(1024, np.float32)
b = rvv.scale(a, 3.0)
```