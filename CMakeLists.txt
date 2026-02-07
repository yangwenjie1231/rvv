cmake_minimum_required(VERSION 3.16)
project(rvv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 读取编译标志
file(STRINGS compile_flags.txt CXX_FLAGS_RAW)
string(REPLACE " " CMAKE_CXX_FLAGS "${CXX_FLAGS_RAW}")
file(STRINGS compile_definitions.txt DEFS_RAW)
foreach(d ${DEFS_RAW})
  add_definitions(-D${d})
endforeach()

# 2. 找 pybind11（假设系统已安装，也可 git submodule）
find_package(pybind11 REQUIRED)

# 3. 源文件
file(GLOB_RECURSE SOURCES src/*.cpp)

# 4. 创建 Python 扩展模块
pybind11_add_module(rvv ${SOURCES})

# 5. 安装到 build/dist 方便打包
install(TARGETS rvv DESTINATION build/dist)

# 6. 生成 whl（调用 setuptools）
add_custom_command(TARGET rvv POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Building whl..."
    COMMAND python3 setup.py bdist_wheel
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Packaging wheel..."
)