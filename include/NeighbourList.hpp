#ifndef NEIGHBOUR_LIST_HPP
#define NEIGHBOUR_LIST_HPP

#include "Atoms.hpp"
#include "config.h"

#include <vector>

struct NeighbourList{
    torch::Tensor source_index;                     //ソース原子のインデックス
    torch::Tensor target_index;                     //ターゲット原子のインデックス
    torch::Tensor NL_config;                        //隣接リスト構築時点での配置を保存しておく配列
    torch::Tensor cutoff;                           //カットオフ距離 (1, )
    torch::Tensor margin;                           //カットオフからのマージン (1, )

    //デバイスの移動
    void to(torch::Device device){
        device_ = device;
        source_index.to(device);
        target_index.to(device);
        NL_config.to(device);
        cutoff.to(device);
        margin.to(device);
    }

    //デバイスの取得
    torch::Device device() { return device_; }

    //NLの作成
    void generate(Atoms atoms){
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
        torch::Tensor rlist2 = (cutoff + margin).pow(2);

        //dist2 < rlist2を満たすなら1(true), 満たさないなら0(false)
        torch::Tensor mask = dist2 < rlist2;

        //i = jを除外
        mask.fill_diagonal_(0);

        //インデックスの取得
        //indices[0]がiのインデックス、indices[1]がjのインデックス
        auto indices = torch::where(mask);
        source_index = indices[0].to(kIntType);
        target_index = indices[1].to(kIntType);

        //各粒子iが持つ隣接粒子の数を計算
        torch::Tensor num_neighbours = mask.sum({1}).to(kIntType);

        NL_config = pos.clone();
    }

    //NLの確認
    void check(const Atoms atoms){
        torch::TensorOptions options = torch::TensorOptions().device(device_);
        torch::Tensor pos = atoms.positions().to(device_);  //位置ベクトル (N, 3)
        torch::Tensor Lbox = atoms.box_size().to(device_);  //シミュレーションボックスの大きさ
        torch::Tensor Linv = 1.0 / Lbox;                    //ボックスの大きさの逆数

        //前回隣接リストを構築した配置と比べて変異が最大の2粒子を探す。
        //距離の計算
        torch::Tensor diff_position = pos - NL_config;  // (N, 3)
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
        if( (max1st + max2nd + 2 * torch::sqrt(max1st * max2nd) > margin * margin).item<bool>() ){
            generate(atoms);
        }
    }

    private:
    torch::Device device_;
};

#endif