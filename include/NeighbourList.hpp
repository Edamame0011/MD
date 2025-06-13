#ifndef NEIGHBOUR_LIST_HPP
#define NEIGHBOUR_LIST_HPP

#include "Atoms.hpp"

#include <vector>

class NeighbourList{
    public:
        //コンストラクタ
        NeighbourList(torch::Tensor cutoff, torch::Tensor margin, torch::Device device = torch::kCPU);

        //ゲッタ
        const torch::Tensor& source_index() const { return source_index_; }
        const torch::Tensor& target_index() const { return target_index_; }
        const torch::Tensor& cutoff() const { return cutoff_; }
        const torch::Tensor& NL_config() const { return NL_config_; }
        const torch::Device& device() const { return device_; }

        //デバイスの移動
        void to(torch::Device device);

        //NLの作成
        void generate(const Atoms& atoms);

        //NLの確認
        void update(const Atoms& atoms);

    private:
    torch::Tensor source_index_;                     //ソース原子のインデックス (num_edges, )
    torch::Tensor target_index_;                     //ターゲット原子のインデックス (num_edges, )
    torch::Tensor NL_config_;                        //隣接リスト構築時点での配置を保存しておく配列
    torch::Tensor cutoff_;                           //カットオフ距離 (1, )
    torch::Tensor margin_;                           //カットオフからのマージン (1, )
    torch::Device device_;
};

#endif