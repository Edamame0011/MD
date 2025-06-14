#include "Atoms.hpp"
#include "config.h"

//コンストラクタ
Atoms::Atoms(std::vector<Atom> atoms, torch::Device device) : device_(device)
{
    std::size_t N = atoms.size();
    std::vector<torch::Tensor> positions;
    std::vector<torch::Tensor> velocities;
    std::vector<torch::Tensor> forces;
    std::vector<torch::Tensor> masses;
    std::vector<torch::Tensor> atomic_numbers;
    for(std::size_t i = 0; i < N; i ++){
        atoms[i].to(device_);
        positions.push_back(atoms[i].position());
        velocities.push_back(atoms[i].velocity());
        forces.push_back(atoms[i].force());
        masses.push_back(atoms[i].mass());
        atomic_numbers.push_back(atoms[i].atomic_number());
    }
    //torch::Tensorに変換
    n_atoms_ = torch::tensor(static_cast<int64_t>(N), torch::TensorOptions().device(device).dtype(kIntType));
    positions_ = torch::stack(positions); 
    velocities_ = torch::stack(velocities); 
    forces_ = torch::stack(forces); 
    masses_ = torch::stack(masses);
    atomic_numbers_ = torch::stack(atomic_numbers);
}

Atoms::Atoms(torch::Device device) : Atoms(std::vector<Atom>(), device)
{
}

//セッタ
void Atoms::set_positions(const torch::Tensor& positions) { 
    //値が不正でないかのチェック
    TORCH_CHECK(positions.size(0) == n_atoms_.item<int64_t>() && positions.size(1) == 3, "positionsの形状は(N, 3)である必要があります。");
    positions_ = positions; 
}
void Atoms::set_velocities(const torch::Tensor& velocities) { 
    TORCH_CHECK(velocities.size(0) == n_atoms_.item<int64_t>() && velocities.size(1) == 3, "velocitiesの形状は(N, 3)である必要があります。");
    velocities_ = velocities; 
}
void Atoms::set_forces(const torch::Tensor& forces) { 
    TORCH_CHECK(forces.size(0) == n_atoms_.item<int64_t>() && forces.size(1) == 3, "forcesの形状は(N, 3)である必要があります。");
    forces_ = forces;
}
void Atoms::set_box_size(const torch::Tensor& box_size){
    TORCH_CHECK(box_size.item<float>() >= 0, "box_sizeは正の数である必要があります。");
    box_size_ = box_size; 
}
void Atoms::set_potential_energy(const torch::Tensor& potential_energy){
    TORCH_CHECK(potential_energy.dim() == 0, "potential_energyの次元は0である必要があります。");
    potential_energy_ = potential_energy;
}

//デバイスの移動
void Atoms::to(torch::Device device) {
    device_ = device;
    positions_ = positions_.to(device);
    velocities_ = velocities_.to(device);
    forces_ = forces_.to(device);
    masses_ = masses_.to(device);
    atomic_numbers_ = atomic_numbers_.to(device);
    n_atoms_ = n_atoms_.to(device);
    potential_energy_ = potential_energy_.to(device);
    box_size_ = box_size_.to(device);
}

//物理量の計算
torch::Tensor Atoms::kinetic_energy() const {
    auto vel_sq = torch::pow(velocities_, 2);
    auto sum_vel_sq = torch::sum(vel_sq, 1);
    auto kinetic_energies = 0.5 * masses_.squeeze() * sum_vel_sq;
    return torch::sum(kinetic_energies);
}

//周期境界条件の補正
void Atoms::apply_pbc(){
    positions_ -= box_size_ * torch::floor(positions_ / box_size_ + 0.5);
}

//周期境界条件の補正（何回移動したかをboxに保存）
void Atoms::apply_pbc(torch::Tensor& box){
    torch::Tensor box_indices = torch::floor(positions_ / box_size_ + 0.5);
    positions_ -= box_size_ * box_indices;
    box += box_indices;
}

//位置の更新
void Atoms::positions_update(const torch::Tensor dt, torch::Tensor& box){
    positions_ += dt * velocities_;
    apply_pbc(box);
}

//速度の更新
void Atoms::velocities_update(const torch::Tensor dt){
    velocities_ += 0.5 * dt * forces_ / masses_;
}