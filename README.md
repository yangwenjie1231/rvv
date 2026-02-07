# rvv – SG2002 RVV 0.7.1 加速库

## 功能
- 向量级：add / sub / scale / dot / norm_l2 / normalize  
- 矩阵级：add2d / scale2d / matmul / transpose / mv（矩阵×向量）  
- 接口 100 % 兼容 NumPy，输入输出均为 `numpy.ndarray`  
- 内部自动使用玄铁 C906 RVV intrinsics，SG2002 实测 4×+ 加速

## 快速安装
```bash
cd rvv
python project.py build
pip install build/dist/*.whl
```

## 用法示例
```python
import numpy as np
import rvv

a = np.random.rand(1024).astype(np.float32)
b = np.random.rand(1024).astype(np.float32)

c = rvv.add(a, b)        # 向量加法
s = rvv.dot(a, b)        # 点积
n = rvv.norm_l2(a)       # L2 范数
```

## 运行测试
```bash
python tests/test_rvv.py
```

## 文档
详见 `docs/` 目录，或直接在 Python 内 `help(rvv.add)` 查看 docstring。