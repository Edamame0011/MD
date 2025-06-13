#ifndef INFERENCE_HPP
#define INFERENCE_HPP

#include "Atoms.hpp"
#include "NeighbourList.hpp"

#include <torch/script.h>
#include <torch/torch.h>

namespace inference{
    torch::jit::script::Module load_model(std::string model_path);                                                                                            //モデルのロード  
    c10::ivalue::TupleElements infer_from_tensor(torch::jit::script::Module& module, torch::Tensor x, torch::Tensor edge_index, torch::Tensor edge_weight);   //グラフの要素（テンソル）からの推論                                                                                                                 
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RadiusInteractionGraph(Atoms& atoms, torch::Tensor cutoff);                                       //cutoff距離以内にある原子のペアを探す
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RadiusInteractionGraph(Atoms& atoms, NeighbourList NL);                                           //隣接リストを使用する場合はこっち
    void calc_energy_and_force_MLP(torch::jit::script::Module& module, Atoms& atoms, torch::Tensor cutoff);                                                   //一つの構造に対して、エネルギーと力を計算
    void calc_energy_and_force_MLP(torch::jit::script::Module& module, Atoms& atoms, NeighbourList NL);                                                       //隣接リストを使用する場合はこっち
}

#endif