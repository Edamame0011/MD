#ifndef MD_HPP
#define MD_HPP

#include "Atoms.hpp"
#include "NeighbourList.hpp"
#include "config.h"

#include <torch/script.h>
#include <torch/torch.h>

class MD{
    public:
        //コンストラクタ
        MD(torch::Tensor dt, torch::Tensor cutoff, torch::Tensor margin, std::string data_path, std::string model_path, torch::Device device = torch::kCPU); 
        MD(RealType dt, RealType cutoff, RealType margin, std::string data_path, std::string model_path, torch::Device device = torch::kCPU); 

        //初期化
        void init_vel_MB(const float float_targ);                       //原子の速度の初期化

        //シミュレーション
        void NVE(const float tsim);
        void NVE_from_grad(const float tsim);                                     //エネルギーだけモデルで推論し力はその微分で求める

    private:
        //その他（補助用関数）
        void print_energies(long t);                                    //結果の出力
        void remove_drift();                                            //全体速度の除去

        //シミュレーション用
        torch::Tensor dt_;                                              //時間刻み幅
        torch::Tensor Lbox_;                                            //シミュレーションセルのサイズ
        torch::Tensor Linv_;                                            //セルのサイズの逆数
        NeighbourList NL_;                                              //隣接リスト

        //MLP用変数
        torch::jit::script::Module module_;                              //モデルを格納する変数

        //系
        Atoms atoms_;                                                    //原子
        torch::Tensor num_atoms_;                                        //原子数

        //シミュレーションデバイス
        torch::Device device_;
};

#endif