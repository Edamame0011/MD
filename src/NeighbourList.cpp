#include "NeighbourList.hpp"
#include "config.h"

#include <stdexcept>

NeighbourList::NeighbourList(torch::Tensor cutoff, torch::Tensor margin, torch::Device device)
              : cutoff_(cutoff), margin_(margin), device_(device) 
{
    //値が不正でないかのチェック
    if(cutoff_.item<float>() <= 0){
        throw std::invalid_argument("cutoff距離は正の数である必要があります。");
    }
    if(margin_.item<float>() <= 0){
        throw std::invalid_argument("margin距離は正の数である必要があります。");
    }
}

//デバイスの移動
void NeighbourList::to(torch::Device device){
    device_ = device;
    source_index_ = source_index_.to(device);
    target_index_ = target_index_.to(device);
    NL_config_ = NL_config_.to(device);
    cutoff_ = cutoff_.to(device);
    margin_ = margin_.to(device);
}

//NLの作成
void NeighbourList::generate(const Atoms& atoms){
    torch::TensorOptions options = torch::TensorOptions().device(device_);
    torch::Tensor pos = atoms.positions().to(device_);  //位置ベクトル (N, 3)
    torch::Tensor Lbox = atoms.box_size().to(device_);  //シミュレーションボックスの大きさ
    torch::Tensor Linv = 1.0 / Lbox;                    //ボックスの大きさの逆数
    //系全体でのijペアの数を数える変数
    torch::Tensor nlist = torch::tensor(-1, options.dtype(kIntType));
    //距離の計算
    //pos.unsqueeze(1) -> (N, 1, 3)
    //pos.unsqueeze(0) -> (1, N, 3)
    torch::Tensor diff_position = pos.unsqueeze(1) - pos.unsqueeze(0); //(N, N, 3)
    //周期境界条件の適用
    diff_position -= Lbox * torch::floor(diff_position * Linv + 0.5);
    //距離の2乗を計算
    //diff_positionの要素を2乗し、dim=2について足す
    torch::Tensor dist2 = torch::sum(diff_position.pow(2), 2); //(N, N)
    //マージンを考慮したカットオフ距離
    torch::Tensor rlist2 = (cutoff_ + margin_).pow(2);
    //dist2 < rlist2を満たすなら1(true), 満たさないなら0(false)
    torch::Tensor mask = dist2 < rlist2;
    //i = jを除外
    mask.fill_diagonal_(0);
    //インデックスの取得
    //indices[0]がiのインデックス、indices[1]がjのインデックス
    auto indices = torch::where(mask);
    source_index_ = indices[0].to(kIntType);
    target_index_ = indices[1].to(kIntType);
    //各粒子iが持つ隣接粒子の数を計算
    torch::Tensor num_neighbours = mask.sum({1}).to(kIntType);
    NL_config_ = pos.clone();
}

void NeighbourList::update(const Atoms& atoms){
    torch::TensorOptions options = torch::TensorOptions().device(device_);
    torch::Tensor pos = atoms.positions().to(device_);  //位置ベクトル (N, 3)
    torch::Tensor Lbox = atoms.box_size().to(device_);  //シミュレーションボックスの大きさ
    torch::Tensor Linv = 1.0 / Lbox;                    //ボックスの大きさの逆
    //前回隣接リストを構築した配置と比べて変異が最大の2粒子を探す。
    //距離の計算
    torch::Tensor diff_position = pos - NL_config_;  // (N, 3)
    //周期境界条件の適用
    diff_position -= Lbox * torch::floor(diff_position * Linv + 0.5);
    //距離の2乗
    torch::Tensor dist2 = torch::sum(diff_position.pow(2), 1);  //(N, )
    //大きい順にソートし、1番目と2番目に大きい距離を取得
    auto sorted_result = torch::sort(dist2, -1, true);
    torch::Tensor sorted_dist2 = std::get<0>(sorted_result);
    torch::Tensor max1st = sorted_dist2[0];
    torch::Tensor max2nd = sorted_dist2[1];
    //移動距離の和がマージンを超えたらNLを作り直す。
    //torch::Tensorのままで比較すると、torch::Tensor型が返ってくるため、item<float>()でfloatに変換してから比較するか、比較した後でitem<bool>()でbool型に変換する。
    if( (max1st + max2nd + 2 * torch::sqrt(max1st * max2nd) > margin_ * margin_).item<bool>() ){
        generate(atoms);
    }
}