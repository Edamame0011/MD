#ifndef CONFIG_H
#define CONFIG_H

#include <torch/torch.h>
#include <torch/script.h>

//精度の設定
using RealType = float;
constexpr torch::ScalarType kRealType = torch::kFloat32;

using IntType = int;
constexpr torch::ScalarType kIntType = torch::kInt64;

//定数
const RealType unit_fs = 1e-15;
//ボルツマン定数 (eV / K)
const torch::Tensor boltzmann_constant = torch::tensor(8.617333262145e-5, torch::TensorOptions().dtype(kRealType));
//変換係数 (ev / amu) -> ((Å / fs) ^ 2)
const torch::Tensor unit_conversion_factor = torch::tensor(103.6427, torch::TensorOptions().dtype(kRealType));

#endif