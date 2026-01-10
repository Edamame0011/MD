#include "MD.hpp"

#include "xyz.hpp"
#include "inference.hpp"
#include "config.h"
#include "LJ.hpp"

//=====コンストラクタ=====
MD::MD(torch::Tensor dt, torch::Tensor cutoff, torch::Tensor margin, std::string data_path, std::string model_path, torch::Device device)
   : dt_(dt), NL_(cutoff, margin, device), device_(device), atoms_(Atoms(device))
{
    //モデルの読み込み
    module_ = inference::load_model(model_path);
    module_.to(device);

    //初期構造のロード
    xyz::load_atoms(data_path, atoms_, device);
    num_atoms_ = atoms_.size();
    Lbox_ = atoms_.box_size();
    Linv_ = 1.0 / Lbox_;

    //周期境界条件の補正
    atoms_.apply_pbc();

    //使用する定数のデバイスを移動しておく。
    boltzmann_constant_ = torch::tensor(boltzmann_constant, torch::TensorOptions().dtype(kRealType).device(device_));
    conversion_factor_ = torch::tensor(conversion_factor, torch::TensorOptions().dtype(kRealType).device(device_));

    //その他の変数の初期化
    t_ = 0;
    temp_ = 0.0;
    dt_real_ = dt_.item<RealType>();
    box_ = torch::zeros({num_atoms_.item<IntType>(), 3}, torch::TensorOptions().dtype(kIntType).device(device_));
    traj_path_ = "./trajectory.xyz";
}

MD::MD(RealType dt, RealType cutoff, RealType margin, std::string data_path, std::string model_path, torch::Device device)
   : MD(torch::tensor(dt, torch::TensorOptions().device(device).dtype(kRealType)), 
        torch::tensor(cutoff, torch::TensorOptions().device(device).dtype(kRealType)), 
        torch::tensor(margin, torch::TensorOptions().device(device).dtype(kRealType)), 
        data_path, model_path, device) {}

MD::MD(RealType dt, RealType cutoff, RealType margin, const Atoms& atoms, torch::Device device) : dt_(torch::tensor(dt, torch::TensorOptions().device(device).dtype(kRealType))), device_(device), atoms_(atoms), NL_(torch::tensor(cutoff, torch::TensorOptions().device(device).dtype(kRealType)), torch::tensor(margin, torch::TensorOptions().device(device).dtype(kRealType)), device) {
    num_atoms_ = atoms_.size();
    Lbox_ = atoms_.box_size();
    Linv_ = 1.0 / Lbox_;

    atoms_.apply_pbc();

    boltzmann_constant_ = torch::tensor(boltzmann_constant, torch::TensorOptions().dtype(kRealType).device(device_));
    conversion_factor_ = torch::tensor(conversion_factor, torch::TensorOptions().dtype(kRealType).device(device_));

    t_ = 0;
    temp_ = 0.0;
    dt_real_ = dt_.item<RealType>();
    box_ = torch::zeros({num_atoms_.item<IntType>(), 3}, torch::TensorOptions().dtype(kIntType).device(device_));
    traj_path_ = "./trajectory.xyz";
}

//=====シミュレーション=====
//保存用の関数をラムダ式で渡しているだけ

//NVEシミュレーション
void MD::NVE(const RealType tsim, const RealType temp, const IntType step, const bool is_save) {
    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)" << std::endl;

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);
    print_energies();

    if (is_save) xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);

    if(is_save) {
        NVE_loop(tsim, temp, [this, step]() {
            if(t_ % step == 0) [[unlikely]] {
                print_energies();

                xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);
            }
        });
    }
    else {
        NVE_loop(tsim, temp, [this, step]() {
            if(t_ % step == 0) [[unlikely]] {
                print_energies();
            }
        });
    }
}

//NVEシミュレーション（logスケールで保存）
void MD::NVE(const RealType tsim, const RealType temp, const std::string log, const bool is_save) {
    if(log != "log") {
        return; 
    }

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)" << std::endl;

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);
    print_energies();

    if (is_save) xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);

    const auto logbin = std::pow(10.0, 1.0 / 9);
    int counter = 5;
    auto checker = 1e-3 * std::pow(logbin, counter);

    double current_time = static_cast<double>(dt_real_) * static_cast<double>(t_);

    //現在の時間に合わせてcheckerを更新
    while (checker <= current_time) {
        checker *= logbin;
    }

    if(is_save) {
        NVE_loop(tsim, temp, [this, &checker, logbin]() {
            if(static_cast<double>(dt_real_) * static_cast<double>(t_) > checker) [[unlikely]] {
                checker *= logbin;
                print_energies();

                xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);
            }
        });
    }
    else {
        NVE_loop(tsim, temp, [this, &checker, logbin]() {
            if(static_cast<double>(dt_real_) * static_cast<double>(t_) > checker) [[unlikely]] {
                checker *= logbin;
                print_energies();
            }
        });
    }
}

//NVTシミュレーション
template <typename ThermostatType>
void MD::NVT(const RealType tsim, ThermostatType& Thermostat, const IntType step, const bool is_save) {
    Thermostat.setup(atoms_);

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)、temperature (K)" << std::endl;

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);
    print_energies();
    if (is_save) xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);

    if(is_save) {
        NVT_loop(tsim, Thermostat, [this, step]() {
            if(t_ % step == 0) [[unlikely]] {
                print_energies();

                xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);
            }
        });
    }
    else {
        NVT_loop(tsim, Thermostat, [this, step]() {
            if(t_ % step == 0) [[unlikely]] {
                print_energies();
            }
        });
    }
}

// NVTシミュレーション（logスケールで保存 + 1 decade あたり N 点 + 各点で M ステップ連続保存, 連続保存時の間隔も指定できる）
// decadeはステップ数で判定
template <typename ThermostatType>
void MD::NVT(const RealType tsim,
             ThermostatType& Thermostat,
             const std::string log,
             const bool is_save,
             const IntType N_per_decade,
             const IntType M_burst,
             const IntType interval_burst)
{
    if (log != "log") {
        return;
    }

    // 熱浴のセットアップ
    Thermostat.setup(atoms_);

    // ログの見出しを出力
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、"
                 "total energy (eV)、temperature (K)" << std::endl;

    // NL の作成
    NL_.generate(atoms_);

    // モデルの推論
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);

    // 初期状態を 1 回出力（run の t=0 に相当）
    print_energies();
    if (is_save) {
        xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);
    }

    // ---- 対数サンプリング + バースト設定（run 相対ステップで管理） ----
    const int N = (N_per_decade > 0) ? static_cast<int>(N_per_decade) : 1;
    const int M = (M_burst > 0) ? static_cast<int>(M_burst) : 0;
    const int interval = (interval_burst > 0) ? static_cast<int>(interval_burst) : 1;
    const double r = std::pow(10.0, 1.0 / static_cast<double>(N));

    const IntType t0 = t_;                  // run 開始時の通算ステップ
    IntType next_step_rel = static_cast<IntType>(1);  // 次のログ点（run 相対ステップ）

    int burst_samples_left = 0; // 残りサンプル点数（M点）
    int burst_wait = 0;         // 次の出力までの待ちステップ

    auto callback = [this, r, t0, &next_step_rel,
                     &burst_samples_left, &burst_wait,
                     M, interval, is_save]() {

        // NVT_loop では step() の後に t_++ して output_action() を呼ぶので、
        // callback が呼ばれる時点の run 相対ステップは 1,2,3,... になる。
        const IntType s_rel = t_ - t0;

        // (A) ログ点の進行（バースト中でも next_step_rel を未来へ進める：スキップ扱い）
        bool hit_log_point = false;
        if (s_rel >= next_step_rel) [[unlikely]] {
            hit_log_point = true;
            while (s_rel >= next_step_rel) {
                const double cand_d = std::ceil(static_cast<double>(next_step_rel) * r);
                const IntType cand = static_cast<IntType>(cand_d);
                next_step_rel = std::max(next_step_rel + static_cast<IntType>(1), cand);
            }
        }

        // (B) ログ点に到達し、かつバースト外ならバースト開始（到達ステップを1点目として即出力）
        if (hit_log_point && burst_samples_left == 0 && M > 0) {
            burst_samples_left = M;
            burst_wait = 0;

            // 1点目（到達ステップ）を即出力
            print_energies();
            if (is_save) {
                xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);
            }
            --burst_samples_left;

            burst_wait = (burst_samples_left > 0) ? interval : 0;
        }

        // (C) バースト中の interval 出力
        if (burst_samples_left > 0) {
            if (burst_wait == 0) {
                print_energies();
                if (is_save) {
                    xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);
                }
                --burst_samples_left;
                burst_wait = (burst_samples_left > 0) ? interval : 0;
            } else {
                --burst_wait;
            }
        }
    };

    NVT_loop(tsim, Thermostat, callback);
}



//温度を変化させながらシミュレーション
template <typename ThermostatType>
void MD::NVT_anneal(const RealType cooling_rate, ThermostatType& Thermostat, const RealType targ_temp, const IntType step, const bool is_save) {
    // ★ ここで現在の運動温度を取得して temp_ に同期
    temp_ = atoms_.temperature().item<RealType>();
    Thermostat.set_temp(temp_);
    Thermostat.setup(atoms_);

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)、temperature (K)" << std::endl;

    print_energies();
    if (is_save) xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);

    if(is_save) {
        NVT_anneal_loop(cooling_rate, Thermostat, targ_temp, [this, step]() {
            if(t_ % step == 0) [[unlikely]] {
                print_energies();

                xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);
            }
        });
    }
    else {
        NVT_anneal_loop(cooling_rate, Thermostat, targ_temp, [this, step]() {
            if(t_ % step == 0) [[unlikely]] {
                print_energies();
            }
        });
    }
}

// 温度変化NVTシミュレーション（logスケールで保存 + 1 decade あたり N 点 + 各点で M ステップ連続保存）
template <typename ThermostatType>
void MD::NVT_anneal(const RealType cooling_rate,
                    ThermostatType& Thermostat,
                    const RealType targ_temp,
                    const std::string log,
                    const bool is_save,
                    const IntType N_per_decade,
                    const IntType M_burst,
                    const IntType interval_burst)
{
    if (log != "log") {
        return;
    }

    // 現在の運動温度を取得して temp_ に同期
    temp_ = atoms_.temperature().item<RealType>();
    Thermostat.set_temp(temp_);
    Thermostat.setup(atoms_);

    // NL の作成
    NL_.generate(atoms_);

    // MLP 推論
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_);

    // ログヘッダ
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、"
                 "total energy (eV)、temperature (K)" << std::endl;

    // 初期出力（run の t=0）
    print_energies();
    if (is_save) {
        xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);
    }

    // ---- 対数サンプリング + バースト設定（run 相対ステップで管理） ----
    const int N = (N_per_decade > 0) ? static_cast<int>(N_per_decade) : 1;
    const int M = (M_burst > 0) ? static_cast<int>(M_burst) : 0;
    const int interval = (interval_burst > 0) ? static_cast<int>(interval_burst) : 1;
    const double r = std::pow(10.0, 1.0 / static_cast<double>(N));

    const IntType t0 = t_;                  // run 開始時の通算ステップ
    IntType next_step_rel = static_cast<IntType>(1);

    int burst_samples_left = 0;
    int burst_wait = 0;

    auto callback = [this, r, t0, &next_step_rel,
                     &burst_samples_left, &burst_wait,
                     M, interval, is_save]() {

        const IntType s_rel = t_ - t0;

        // (A) ログ点の進行（スキップ扱いで前進）
        bool hit_log_point = false;
        if (s_rel >= next_step_rel) [[unlikely]] {
            hit_log_point = true;
            while (s_rel >= next_step_rel) {
                const double cand_d = std::ceil(static_cast<double>(next_step_rel) * r);
                const IntType cand = static_cast<IntType>(cand_d);
                next_step_rel = std::max(next_step_rel + static_cast<IntType>(1), cand);
            }
        }

        // (B) バースト開始（到達ステップを1点目として即出力）
        if (hit_log_point && burst_samples_left == 0 && M > 0) {
            burst_samples_left = M;
            burst_wait = 0;

            print_energies();
            if (is_save) {
                xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);
            }
            --burst_samples_left;

            burst_wait = (burst_samples_left > 0) ? interval : 0;
        }

        // (C) バースト中の interval 出力
        if (burst_samples_left > 0) {
            if (burst_wait == 0) {
                print_energies();
                if (is_save) {
                    xyz::save_unwrapped_atoms(traj_path_, atoms_, box_);
                }
                --burst_samples_left;
                burst_wait = (burst_samples_left > 0) ? interval : 0;
            } else {
                --burst_wait;
            }
        }
    };

    NVT_anneal_loop(cooling_rate, Thermostat, targ_temp, callback);
}

//=====シミュレーション（1ステップ）=====
//NVEの1ステップ
void MD::step() {
    atoms_.velocities_update(dt_);      //速度の更新（1回目）
    atoms_.positions_update(dt_, box_);  //位置の更新
    NL_.update(atoms_);                 //NLの確認と更新
    inference::calc_energy_and_force_MLP(module_, atoms_, NL_); //力の更新
    atoms_.velocities_update(dt_);      //速度の更新（2回目）
}

//NVTの1ステップ（Nose-Hoover）
void MD::step(NoseHooverThermostat& Thermostat) {
    Thermostat.update(atoms_, dt_);     //熱浴の更新
    step();
    Thermostat.update(atoms_, dt_);     //熱浴の更新
}

//NVTの1ステップ（Bussi）
void MD::step(BussiThermostat& Thermostat) {
    step();
    Thermostat.update(atoms_, dt_);     //熱浴の更新
}

//=====シミュレーション（メインループ）=====
template <typename OutputAction>
void MD::NVE_loop(const RealType tsim, const RealType temp, OutputAction output_action) {
    IntType steps = tsim / dt_real_;    //総ステップ数
    steps += t_;

    while(t_ < steps){
        step();

        t_ ++;

        //出力
        output_action();
    }
}

template <typename OutputAction, typename ThermostatType>
void MD::NVT_loop(const RealType tsim, ThermostatType& Thermostat, OutputAction output_action) {
    IntType steps = tsim / dt_real_;    //総ステップ数
    steps += t_;

    while(t_ < steps){
        step(Thermostat);
        t_ ++;

        //出力
        output_action();

        //ドリフト速度の除去
        if(!(t_ & 127)) { 
            atoms_.remove_drift();
        }
    }
}

template <typename OutputAction, typename ThermostatType>
void MD::NVT_anneal_loop(const RealType cooling_rate, ThermostatType& Thermostat, const RealType targ_temp, OutputAction output_action) {
    RealType dT = cooling_rate * dt_real_;                                         //1ステップあたりの下降温度
    IntType quench_steps = static_cast<IntType>(std::ceil((temp_ - targ_temp) / dT));    //冷却ステップ数

    if (temp_ < targ_temp) {
        quench_steps = -quench_steps;
        dT = -dT;
    }

    quench_steps += t_;

    //冷却
    //ANNEALの場合はここでtemp_による制御が入っているから、
    //最初の段階でtemp_を初期化する必要があった。
    while(t_ < quench_steps){
        temp_ -= dT;
        Thermostat.set_temp(temp_);
        step(Thermostat);
        t_ ++;

        //出力
        output_action();

        //ドリフト速度の除去
        if(!(t_ & 127)) { 
            atoms_.remove_drift();
        }
    }

    temp_ = targ_temp;
    Thermostat.set_temp(targ_temp);
}

//=====その他=====
//速度（温度）の初期化
void MD::init_temp(const RealType initial_temp){
    temp_ = initial_temp;

    //平均0、分散1のランダムな分布を作成
    torch::Tensor velocities = torch::randn({num_atoms_.item<int64_t>(), 3}, torch::TensorOptions().device(device_).dtype(kRealType));

    //分散を√(k_B * T / m)にする。
    //この時、(eV / amu) -> ((Å / fs) ^ 2)
    torch::Tensor masses = atoms_.masses();
    torch::Tensor temp = torch::tensor(temp_, torch::TensorOptions().dtype(kRealType).device(device_));
    torch::Tensor sigma = torch::sqrt((boltzmann_constant_ * temp_ * conversion_factor_) / masses);
    //velocitiesにsigmaを掛けることで分散を調節。
    //この時、velocities (N, 3)とsigma (N, )を計算するために、sigma (N, ) -> (N, 1)
    velocities *= sigma.unsqueeze(1);

    //全体速度の除去
    torch::Tensor drift_velocity = torch::mean(velocities, 0);
    velocities -= drift_velocity;

    atoms_.set_velocities(velocities);
}

//エネルギーの出力
void MD::print_energies(){
    RealType K = atoms_.kinetic_energy().item<RealType>();
    RealType U = atoms_.potential_energy().item<RealType>();
    RealType temperature = atoms_.temperature().item<RealType>();
    
    //時刻、運動エネルギー、ポテンシャルエネルギー、全エネルギー、温度を出力
    std::cout << std::setprecision(15) << std::scientific << dt_.item<RealType>() * t_ << "," 
                                                          << K << "," 
                                                          << U << "," 
                                                          << K + U << "," 
                                                          << temperature << std::endl;
}

void MD::reset_step() {
    t_ = 0;
}

void MD::reset_box() {
    box_ = torch::zeros({num_atoms_.item<IntType>(), 3}, torch::TensorOptions().dtype(kIntType).device(device_));
}

void MD::save_atoms(const std::string& save_path) {
    xyz::save_atoms(save_path, atoms_);
}

void MD::save_unwrapped_atoms(const std::string& save_path) {
    xyz::save_unwrapped_atoms(save_path, atoms_, box_);
}

void MD::set_traj_path(const std::string& path) {
    traj_path_ = path;
}

//=====LJユニットによるテスト用関数=====
//NVEの1ステップ
void MD::step_LJ(torch::Tensor& box) {
    atoms_.velocities_update(dt_);      //速度の更新（1回目）
    atoms_.positions_update(dt_, box);  //位置の更新
    NL_.update(atoms_);                 //NLの確認と更新
    LJ::calc_energy_and_force(atoms_, NL_); //力の更新
    atoms_.velocities_update(dt_);      //速度の更新（2回目）
}

//NVTの1ステップ（Nose-Hoover）
void MD::step_LJ(torch::Tensor& box, NoseHooverThermostat& Thermostat) {
    Thermostat.update(atoms_, dt_);     //熱浴の更新
    step_LJ(box);
    Thermostat.update(atoms_, dt_);     //熱浴の更新
}

//NVTの1ステップ（Bussi）
void MD::step_LJ(torch::Tensor& box, BussiThermostat& Thermostat) {
    step_LJ(box);
    Thermostat.update(atoms_, dt_);     //熱浴の更新
}

template <typename OutputAction>
void MD::NVE_loop_LJ(const RealType tsim, const RealType temp, OutputAction output_action) {
    torch::TensorOptions options = torch::TensorOptions().device(device_);

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)" << std::endl;

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    LJ::calc_energy_and_force(atoms_, NL_);

    IntType steps = tsim / dt_real_;    //総ステップ数
    steps += t_;
    print_energies();

    while(t_ < steps){
        step_LJ(box_);

        t_ ++;

        //出力
        output_action();
    }
}

template <typename OutputAction, typename ThermostatType>
void MD::NVT_loop_LJ(const RealType tsim, ThermostatType& Thermostat, OutputAction output_action) {
    torch::TensorOptions options = torch::TensorOptions().device(device_);

    Thermostat.setup(atoms_);

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)、temperature (K)" << std::endl;

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    LJ::calc_energy_and_force(atoms_, NL_);

    IntType steps = tsim / dt_real_;    //総ステップ数
    steps += t_;
    print_energies();

    while(t_ < steps){
        step_LJ(box_, Thermostat);
        t_ ++;

        //出力
        output_action();

        //ドリフト速度の除去
        if(!(t_ & 127)) { 
            atoms_.remove_drift();
        }
    }
}

template <typename OutputAction, typename ThermostatType>
void MD::NVT_anneal_loop_LJ(const RealType cooling_rate, ThermostatType& Thermostat, const RealType targ_temp, OutputAction output_action) {
    torch::TensorOptions options = torch::TensorOptions().device(device_);

    Thermostat.setup(atoms_);

    //NLの作成
    NL_.generate(atoms_);

    //モデルの推論
    LJ::calc_energy_and_force(atoms_, NL_);

    const RealType dT = cooling_rate * dt_real_;                                            //1ステップあたりの下降温度
    IntType quench_steps = static_cast<IntType>(std::ceil((temp_ - targ_temp) / dT));       //冷却ステップ数
    quench_steps += t_;

    //ログの見出しを出力しておく
    std::cout << "time (fs)、kinetic energy (eV)、potential energy (eV)、total energy (eV)、temperature (K)" << std::endl;

    print_energies();

    //冷却
    while(t_ < quench_steps){
        temp_ -= dT;
        Thermostat.set_temp(temp_);
        step_LJ(box_, Thermostat);
        t_ ++;

        //出力
        output_action();

        //ドリフト速度の除去
        if(!(t_ & 127)) { 
            atoms_.remove_drift();
        }
    }

    temp_ = targ_temp;
    Thermostat.set_temp(targ_temp);
}

//NVEシミュレーション
void MD::NVE_LJ(const RealType tsim, const RealType temp, const IntType step, const bool is_save, const std::string output_path) {
    if(is_save) {
        NVE_loop_LJ(tsim, temp, [this, step]() {
            if(t_ % step == 0) {
                print_energies();

                std::string path = "./outputs/output_traj.xyz";
                xyz::save_unwrapped_atoms(path, atoms_, box_);
            }
        });
    }
    else {
        NVE_loop_LJ(tsim, temp, [this, step]() {
            if(t_ % step == 0) {
                print_energies();
            }
        });
    }

    xyz::save_atoms(output_path, atoms_);
}

//NVTシミュレーション
template <typename ThermostatType>
void MD::NVT_LJ(const RealType tsim, ThermostatType& Thermostat, const IntType step, const bool is_save, const std::string output_path) {
    if(is_save) {
        NVT_loop_LJ(tsim, Thermostat, [this, step]() {
            if(t_ % step == 0) {
                print_energies();

                std::string path = "./outputs/output_traj.xyz";
                xyz::save_unwrapped_atoms(path, atoms_, box_);
            }
        });
    }
    else {
        NVT_loop_LJ(tsim, Thermostat, [this, step]() {
            if(t_ % step == 0) {
                print_energies();
            }
        });
    }

    xyz::save_atoms(output_path, atoms_);
}

//NVTシミュレーション（logスケールで保存）
template <typename ThermostatType>
void MD::NVT_LJ(const RealType tsim, ThermostatType& Thermostat, const std::string log, const bool is_save, const std::string output_path) {
    if(log != "log") {
        return; 
    }

    const auto logbin = std::pow(10.0, 1.0 / 9);
    int counter = 5;
    auto checker = 1e-3 * std::pow(logbin, counter);

    if(is_save) {
        NVT_loop_LJ(tsim, Thermostat, [this, &checker, logbin]() {
            if(static_cast<double>(dt_real_) * static_cast<double>(t_) > checker) {
                checker *= logbin;
                print_energies();

                std::string path = "./outputs/output_traj.xyz";
                xyz::save_unwrapped_atoms(path, atoms_, box_);
            }
        });
    }
    else {
        NVT_loop_LJ(tsim, Thermostat, [this, &checker, logbin]() {
            if(static_cast<double>(dt_real_) * static_cast<double>(t_) > checker) {
                checker *= logbin;
                print_energies();
            }
        });
    }

    xyz::save_atoms(output_path, atoms_);
}
