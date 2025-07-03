#include "inference.hpp"
#include "xyz.hpp"
#include "config.h"

int main(){
    //グローバル変数
    std::vector<Atoms> structures;
    torch::jit::script::Module module;

    //デバイス
    const torch::Device device = torch::kCPU;
    const torch::TensorOptions options = torch::TensorOptions().device(device);

    //定数
    const torch::Tensor cutoff = torch::tensor(5.0, options.dtype(kRealType));
    
    //パス
    const std::string model_path = "../models/deployed_model.pt";
    const std::string data_path = "../data/multiple_structures.xyz";

    //モデルの読み込み
    module = inference::load_model(model_path);

    //構造の読み込み
    xyz::load_structures(data_path, structures, device);

    //一つ一つの構造に対して、エネルギーと力を計算
    for(std::size_t i = 0; i < structures.size(); i ++){
        inference::calc_energy_and_force_MLP(module, structures[i], cutoff);

        //結果の表示
        std::cout << "structure[" << i << "]" << std::endl;
        std::cout << std::setprecision(15) << std::scientific << "potential_energy: " << structures[i].potential_energy() << std::endl;

        torch::Tensor forces = structures[i].forces();
        // 各原子の力をループで表示
        for (int j = 0; j < forces.size(0); ++j) {
            std::cout << "force[" << j + 1 << "]: "
                      << forces[j][0].item<RealType>() << ", "
                      << forces[j][1].item<RealType>() << ", "
                      << forces[j][2].item<RealType>() << std::endl;
        }
        std::cout << std::endl; 
    }

    return 0;
}