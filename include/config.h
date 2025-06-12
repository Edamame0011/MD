#ifndef CONFIG_H
#define CONFIG_H

#include <torch/torch.h>
#include <torch/script.h>

//精度の設定
using RealType = float;
constexpr torch::ScalarType kRealType = torch::kFloat32;

using IntType = int;
constexpr torch::ScalarType kIntType = torch::kInt64;

#endif