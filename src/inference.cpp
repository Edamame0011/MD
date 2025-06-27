#include "Atoms.hpp"
#include "inference.hpp"
#include "config.h"

#include <vector>
#include <array>
#include <iostream>
#include <string>
#include <cctype>
#include <algorithm>
#include <iomanip>

#include <torch/script.h>
#include <torch/torch.h>

torch::jit::script::Module inference::load_model(std::string model_path){
    try{
        torch::jit::script::Module module = torch::jit::load(model_path);
        std::cout << "モデルをロードしました：" << model_path << std::endl; 
        return module; 
    }
    catch(c10::Error& e){
        std::cerr << "モデルの読み込みに失敗しました。" << std::endl
                  << e.what() << std::endl;
        throw;
    }
}

//グラフの要素（テンソル）からの推論
c10::ivalue::TupleElements inference::infer_from_tensor(torch::jit::script::Module& module, torch::Tensor x, torch::Tensor edge_index, torch::Tensor edge_weight){
    //モデルの推論
    module.eval();
    try{
        auto result_iv = module.forward({x, edge_index, edge_weight});
        auto result_tuple = result_iv.toTuple();

        auto elements = result_tuple->elements();

        return elements;
    }
    catch(const c10::Error& e){
        std::cerr << "モデルの推論に失敗しました。" << std::endl
                  << e.what() << std::endl;
        throw;
    }
}

//cutoff距離以内にある原子のペアを探す
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> inference::RadiusInteractionGraph(Atoms& atoms, torch::Tensor cutoff){
    torch::Tensor pos = atoms.positions();
    const torch::Tensor Lbox = atoms.box_size();
    const torch::Tensor Linv = 1 / Lbox;
    
    //距離の計算
    torch::Tensor diff_position = pos.unsqueeze(1) - pos.unsqueeze(0);  //(N, N, 3)
    diff_position -= Lbox * torch::floor(diff_position * Linv + 0.5);
    
    //距離の2乗
    torch::Tensor dist2 = torch::sum(diff_position.pow(2), 2);  //(N, N)

    //マスク
    torch::Tensor mask = dist2 < cutoff.pow(2);
    //i = jを除外
    mask.fill_diagonal_(0);

    //インデックスの取得
    auto indices = torch::where(mask);
    torch::Tensor source_index = indices[0].to(kIntType);   //(num_edges, )
    torch::Tensor target_index = indices[1].to(kIntType);   //(num_edges, )

    //インデックスを一つのtorch::Tensorにまとめる
    torch::Tensor edge_index = torch::stack({source_index, target_index});

    //距離ベクトルの作成
    //2つのインデックスの組み合わせからインデックスを取得
    torch::Tensor distance_vectors = - diff_position.index({source_index, target_index}); //(num_edges, 3)

    //各原子の原子番号を取得
    torch::Tensor x = atoms.atomic_numbers();

    //c++では複数の戻り値を返すことができないため、std::tupleを用いる。
    return std::make_tuple(x, edge_index, distance_vectors);
}

//NLを使う場合
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> inference::RadiusInteractionGraph(Atoms& atoms, NeighbourList NL){
    torch::Tensor pos = atoms.positions();
    const torch::Tensor Lbox = atoms.box_size();
    const torch::Tensor Linv = 1 / Lbox;

    //読み込んだインデックスの原子から、距離を計算
    const torch::Tensor source_pos = pos.index({NL.source_index()});
    const torch::Tensor target_pos = pos.index({NL.target_index()});

    //距離の計算
    torch::Tensor diff_pos_vec = source_pos - target_pos;
    diff_pos_vec -= Lbox * torch::floor(diff_pos_vec * Linv + 0.5);

    //実際のカットオフ距離でフィルタリング
    torch::Tensor dist2 = torch::sum(diff_pos_vec.pow(2), 1);
    torch::Tensor cutoff2 = NL.cutoff().pow(2);

    torch::Tensor mask = torch::lt(dist2, cutoff2); //dist2 < cutoff2

    torch::Tensor source_index = NL.source_index().index({mask});
    torch::Tensor target_index = NL.target_index().index({mask});
    torch::Tensor distance_vectors = - diff_pos_vec.index({mask});

    //インデックスを一つのtorch::Tensorにまとめる
    torch::Tensor edge_index = torch::stack({source_index, target_index});

    //各原子の原子番号を取得
    torch::Tensor x = atoms.atomic_numbers();
    
    return std::make_tuple(x, edge_index, distance_vectors);
}

//一つの構造に対して、エネルギーと力を計算
void inference::calc_energy_and_force_MLP(torch::jit::script::Module& module, Atoms& atoms, torch::Tensor cutoff){
    //グラフ構造を保存する変数
    torch::Tensor x, edge_index, edge_weight;

    //原子をグラフに変換
    std::tie(x, edge_index, edge_weight) = RadiusInteractionGraph(atoms, cutoff);

    //推論
    auto result = infer_from_tensor(module, x, edge_index, edge_weight);

    //力を各原子にセット
    torch::Tensor forces = result[1].toTensor().to(kRealType);
    atoms.set_forces(forces);

    //ポテンシャルをセット
    torch::Tensor energy = result[0].toTensor().to(kRealType);
    atoms.set_potential_energy(energy);
}

//隣接リストを使う場合
void inference::calc_energy_and_force_MLP(torch::jit::script::Module& module, Atoms& atoms, NeighbourList NL){
    //グラフ構造を保存する変数
    torch::Tensor x, edge_index, edge_weight;

    //原子をグラフに変換
    std::tie(x, edge_index, edge_weight) = RadiusInteractionGraph(atoms, NL);

    //推論
    auto result = infer_from_tensor(module, x, edge_index, edge_weight);

    //力を各原子にセット
    torch::Tensor forces = result[1].toTensor().to(kRealType).detach(); //メモリ不足対策に、detach()して、計算グラフから切り離す。
    atoms.set_forces(forces);

    //ポテンシャルをセット
    torch::Tensor energy = result[0].toTensor().to(kRealType).detach(); //メモリ不足対策に、detach()して、計算グラフから切り離す。
    atoms.set_potential_energy(energy);
}

//エネルギーのみをMLPを用いて計算し、力をその微分から求める
void inference::infer_energy_with_MLP_and_clac_force(torch::jit::script::Module& module, Atoms& atoms, NeighbourList NL){
    torch::Tensor x, edge_index, edge_weight;
    std::tie(x, edge_index, edge_weight) = RadiusInteractionGraph(atoms, NL);
    //後で微分を使うためedge_weightのrequires_gradをtrueにする
    edge_weight.requires_grad_(true);
    //推論
    auto result = infer_from_tensor(module, x, edge_index, edge_weight);
    //ポテンシャルを取得
    torch::Tensor energy = result[0].toTensor().to(kRealType);
    
    //力を計算
    //torch::autograd::grad()の引数、戻り値はtorch::TensorList
    torch::Tensor diff_ij = torch::autograd::grad({energy}, {edge_weight})[0];
    //(N, 3)のゼロテンソルを作成
    torch::Tensor force_i = torch::zeros({x.size(0), 3}, torch::TensorOptions().dtype(kRealType));
    torch::Tensor force_j = torch::zeros({x.size(0), 3}, torch::TensorOptions().dtype(kRealType));
    //diff_ijを加算
    force_i.index_add_(0, edge_index[0], diff_ij);
    force_j.index_add_(0, edge_index[1], -diff_ij);

    torch::Tensor force = force_i + force_j;

    energy = energy.detach();
    force = force.detach();

    //力とポテシャルを原子にセット
    atoms.set_potential_energy(energy);
    atoms.set_forces(force);
}