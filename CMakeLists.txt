cmake_minimum_required(VERSION 3.10)
project(MD_MLP)

# C++17 を使用
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include ディレクトリをインクルードパスに追加
include_directories(${PROJECT_SOURCE_DIR}/include)

# ============================================================
# Fix: CUDA::nvToolsExt (NVTX) for conda PyTorch + older CMake
#   - Avoids erroneous -lCUDA::nvToolsExt
#   - Links against conda-provided libnvToolsExt.so.1
# ============================================================
if(DEFINED ENV{CONDA_PREFIX})
  # あなたの環境で見つかった場所（必要なら Python 版だけ調整）
  set(NVTX_LIB_DIR "$ENV{CONDA_PREFIX}/lib/python3.13/site-packages/nvidia/nvtx/lib")
  set(NVTX_LIB "${NVTX_LIB_DIR}/libnvToolsExt.so.1")

  if(EXISTS "${NVTX_LIB}")
    # CUDA::nvToolsExt が未定義なら定義（TorchConfig が要求する可能性がある）
    if(NOT TARGET CUDA::nvToolsExt)
      add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
      set_target_properties(CUDA::nvToolsExt PROPERTIES
        IMPORTED_LOCATION "${NVTX_LIB}"
      )
    endif()

    # 実行時に確実に見つかるように RPATH を設定（ビルドツリー実行でもOK）
    set(CMAKE_BUILD_RPATH "${NVTX_LIB_DIR}")
    set(CMAKE_INSTALL_RPATH "${NVTX_LIB_DIR}")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    message(STATUS "Using conda NVTX: ${NVTX_LIB}")
  else()
    message(WARNING "Conda NVTX not found at: ${NVTX_LIB} (CUDA::nvToolsExt may fail)")
  endif()
else()
  message(WARNING "CONDA_PREFIX is not set; CUDA::nvToolsExt fix is skipped.")
endif()

# ============================================================
# libtorch via conda
# ============================================================
set(LIBTORCH_PATH "$ENV{CONDA_PREFIX}/lib/python3.13/site-packages/torch")
set(Torch_DIR "${LIBTORCH_PATH}/share/cmake/Torch")
find_package(Torch REQUIRED PATHS ${Torch_DIR})

# ============================================================
# cuDNN via conda
# ============================================================
set(CUDNN_ROOT "$ENV{CONDA_PREFIX}")

find_path(CUDNN_INCLUDE_DIR cudnn.h
          PATHS ${CUDNN_ROOT}/include)

find_library(CUDNN_LIBRARY cudnn
             PATHS ${CUDNN_ROOT}/lib ${CUDNN_ROOT}/lib64)

if (CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
  message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
  set(USE_CUDNN 1)
else()
  message(WARNING "cuDNN not found. Compiling without cuDNN support.")
  set(USE_CUDNN 0)
endif()

# ソースファイルのリスト
set(SOURCES
  src/main.cpp
  src/NoseHooverThermostat.cpp
  src/BussiThermostat.cpp
  src/Atoms.cpp
  src/Atom.cpp
  src/NeighbourList.cpp
  src/xyz.cpp
  src/inference.cpp
  src/LJ.cpp
  src/ConfigReader.cpp
)

# 実行ファイルを作成
add_executable(MD_MLP ${SOURCES})

# include（cuDNN）
if (CUDNN_INCLUDE_DIR)
  target_include_directories(MD_MLP PRIVATE ${CUDNN_INCLUDE_DIR})
endif()

# libtorch + cuDNN をリンク
target_link_libraries(MD_MLP PRIVATE ${TORCH_LIBRARIES} ${CUDNN_LIBRARY})

# 念のため明示的にリンク（Torch 側が要求する場合に備える）
if(TARGET CUDA::nvToolsExt)
  target_link_libraries(MD_MLP PRIVATE CUDA::nvToolsExt)
endif()

# （CUDA を明示的に有効化していない場合、以下のCUDA用オプションは基本的に効きません）
target_compile_options(MD_MLP PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:
    --expt-relaxed-constexpr
    -Xcompiler=-O3,-fPIC
  >
)

# このプロパティも CUDA 言語を有効化していないと意味が薄い点に注意
set_target_properties(MD_MLP PROPERTIES CUDA_ARCHITECTURES 89)
set_property(TARGET MD_MLP PROPERTY POSITION_INDEPENDENT_CODE ON)
