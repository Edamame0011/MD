cmake_minimum_required(VERSION 3.10)
project(MD_MLP)

# C++17 を使用（必要に応じて変更）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include ディレクトリをインクルードパスに追加
include_directories(${PROJECT_SOURCE_DIR}/include)

# libtorchの場所
set(LIBTORCH_PATH "/Users/nozawarikuta/libtorch")
set(Torch_DIR "${LIBTORCH_PATH}/share/cmake/Torch")

# libtorchのCMakeモジュールを探す
find_package(Torch REQUIRED PATHS ${Torch_DIR})

# ソースファイルのリスト
set(SOURCES
    src/main_MD.cpp
    src/MD.cpp
    src/Atom.cpp
    src/Atoms.cpp
    src/inference.cpp
    src/load_file.cpp
    src/NeighbourList.cpp
)

# 実行ファイルを作成
add_executable(MD_MLP ${SOURCES})

# libtorchをリンク
target_link_libraries(MD_MLP ${TORCH_LIBRARIES})

# 必須：位置独立コード（libtorchで必要）
set_property(TARGET MD_MLP PROPERTY POSITION_INDEPENDENT_CODE ON)