#include "MD.hpp"
#include "config.h"

int main(){
    //デバイス
    torch::Device device = torch::kCPU;
    torch::TensorOptions options = torch::TensorOptions().device(device);

    //定数
    const RealType dt = 1e-3;
    const RealType cutoff = 5.0;
    const RealType margin = 1.0;

    //パス
    const std::string data_path = "../data/initial_structure.xyz";
    const std::string model_path = "../models/deployed_model.pt";

    //MDオブジェクトの実体化
    MD md = MD(dt, cutoff, margin, data_path, model_path, device);

    //速度の初期化
    md.init_vel_MB(1.0);

    //シミュレーションの開始
    md.NVE(1e+5);
}