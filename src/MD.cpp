#include "MD.hpp"

#include "load_file.hpp"
#include "inference.hpp"
#include "config.h"

//コンストラクタ
MD::MD(torch::Tensor dt, torch::Tensor cutoff, torch::Tensor margin, std::string data_path, std::string model_path, torch::Device device)
   : dt_(dt), atoms_(device), NL_(cutoff, margin, device), device_(device)
{
    //モデルの読み込み
    module_ = inference::load_model(model_path);
    module_.to(device);

    //初期構造のロード
    load_file::load_xyz_atoms(data_path, atoms_, device);
    num_atoms_ = atoms_.size();
    Lbox_ = atoms_.box_size();
    Linv_ = 1.0 / Lbox_;

    //周期境界条件の補正
    atoms_.apply_pbc();

    //使用する定数のデバイスを移動しておく。
    boltzmann_constant.to(device);
    unit_conversion_factor.to(device);
}

MD::MD(RealType dt, RealType cutoff, RealType margin, std::string data_path, std::string model_path, torch::Device device)
   : MD(torch::tensor(dt, torch::TensorOptions().device(device).dtype(kRealType)), 
        torch::tensor(cutoff, torch::TensorOptions().device(device).dtype(kRealType)), 
        torch::tensor(margin, torch::TensorOptions().device(device).dtype(kRealType)), 
        data_path, model_path, device) {}


//速度の初期化
void MD::init_vel_MB(const RealType float_targ){
    //平均0、分散1のランダムな分布を作成
    torch::Tensor velocities = torch::randn({num_atoms_.item<int64_t>(), 3}, torch::TensorOptions().device(device_).dtype(kRealType));

    //分散を√(k_B * T / m)にする。
    //この時、(eV / amu) -> ((Å / fs) ^ 2)のために、unit_conversion_factorを掛ける。
    torch::Tensor masses = atoms_.masses();
    torch::Tensor temp = torch::tensor(float_targ, torch::TensorOptions().dtype(kRealType).device(device_));
    torch::Tensor sigma = torch::sqrt((boltzmann_constant * float_targ * unit_conversion_factor) / masses);
    //velocitiesにsigmaを掛けることで分散を調節。
    //この時、velocities (N, 3)とsigma (N, )を計算するために、sigma (N, ) -> (N, 1)
    velocities *= sigma.unsqueeze(1);

    //全体速度の除去
    torch::Tensor drift_velocity = torch::mean(velocities, 0);
    velocities -= drift_velocity;

    atoms_.set_velocities(velocities);
}

//シミュレーション
void MD::NVE(const float tsim) {
    torch::TensorOptions options = torch::TensorOptions().device(device_);

    //ログの見出しを出力しておく
    std::cout << "time (s)、kinetic energy (eV)、potential energy (eV)、total energy (eV)" << std::endl;

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);

    //周期境界条件のもとで、何個目の箱のミラーに位置しているのかを保存する変数 (N, 3)
    torch::Tensor box = torch::zeros({num_atoms_.item<IntType>(), 3}, options.dtype(kIntType));

    long t = 0; //現在のステップ数
    const long steps = tsim / dt_.item<RealType>();    //総ステップ数
    print_energies(t);

    while(t < steps){
        atoms_.velocities_update(dt_);      //速度の更新（1回目）
        atoms_.positions_update(dt_, box);  //位置の更新
        NL_.update(atoms_);                 //NLの確認と更新
        inference::calc_energy_and_force_MLP(module_, atoms_, NL_); //力の更新
        atoms_.velocities_update(dt_);      //速度の更新（2回目）

        t ++;

        //出力
        //とりあえず100ステップごとに出力
        if(t % 100 == 0){
            print_energies(t);
        }
    }
}

//-----補助用関数-----
//エネルギーの出力
void MD::print_energies(long t){
    RealType K = atoms_.kinetic_energy().item<RealType>();
    RealType U = atoms_.potential_energy().item<RealType>();
    
    //時刻、1粒子当たりの運動エネルギー、1粒子当たりのポテンシャルエネルギー、1粒子当たりの全エネルギーを出力
    std::cout << std::setprecision(15) << std::scientific << dt_.item<RealType>() * t << "," 
                                                          << K << "," 
                                                          << U << "," 
                                                          << K + U << std::endl;
}