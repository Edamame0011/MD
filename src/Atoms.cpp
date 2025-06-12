#include "Atoms.hpp"
#include "config.h"

//コンストラクタ
Atoms::Atoms(std::vector<Atom> atoms, torch::Device device) : device_(device)
{
    n_atoms_ = atoms.size();
    std::vector<torch::Tensor> positions;
    std::vector<torch::Tensor> velocities;
    std::vector<torch::Tensor> forces;
    std::vector<torch::Tensor> masses;
    std::vector<torch::Tensor> atomic_numbers;
    for(std::size_t i = 0; i < n_atoms_; i ++){
        atoms[i].to(device_);
        positions.push_back(atoms[i].position());
        velocities.push_back(atoms[i].velocity());
        forces.push_back(atoms[i].force());
        masses.push_back(atoms[i].mass());
        atomic_numbers.push_back(atoms[i].atomic_number());
    }
    //torch::Tensorに変換
    positions_ = torch::stack(positions); 
    velocities_ = torch::stack(velocities); 
    forces_ = torch::stack(forces); 
    masses_ = torch::stack(masses);
    atomic_numbers_ = torch::stack(atomic_numbers);
}

Atoms::Atoms(torch::Device device) : Atoms(std::vector<Atom>(), device)
{
}

//物理量の計算
torch::Tensor Atoms::kinetic_energy() const {
    auto vel_sq = torch::pow(velocities_, 2);
    auto sum_vel_sq = torch::sum(vel_sq, 1);
    auto kinetic_energies = 0.5 * masses_.squeeze() * sum_vel_sq;
    return torch::sum(kinetic_energies);
}