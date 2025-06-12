// Atoms.hpp
#ifndef ATOMS_HPP
#define ATOMS_HPP

#include "Atom.hpp"
#include <torch/torch.h>
#include <vector>

class Atoms {
public:
    // コンストラクタ
    Atoms(torch::Device device);
    Atoms(std::vector<Atom> atoms, torch::Device device);

    // デバイス移動
    void to(torch::Device device);

    //ゲッタ
    const torch::Tensor& positions() const { return positions_; }
    const torch::Tensor& velocities() const { return velocities_; }
    const torch::Tensor& forces() const { return forces_; }
    const torch::Tensor& atomic_numbers() const { return atomic_numbers_; }
    const torch::Tensor& masses() const { return masses_; }
    const torch::Tensor& box_size() const { return box_size_; }
    const torch::Device& device() const { return device_; }
    std::size_t size() const { return n_atoms_; }

    //セッタ
    void set_positions(const torch::Tensor& positions) { positions_ = positions; }
    void set_velocities(const torch::Tensor& velocities) { velocities_ = velocities; }
    void set_forces(const torch::Tensor& forces) { forces_ = forces; }
    void set_box_size(const torch::Tensor& box_size) { box_size_ = box_size; }

    //物理量の計算
    torch::Tensor kinetic_energy() const;
    torch::Tensor potential_energy() const { return potential_energy_; }
    void set_potential_energy(const torch::Tensor& pe) { potential_energy_ = pe; }

private:
    torch::Device device_;

    //各原子のデータ
    torch::Tensor positions_;      
    torch::Tensor velocities_;     
    torch::Tensor forces_;         
    torch::Tensor masses_;         
    torch::Tensor atomic_numbers_; 

    //系のデータ
    std::size_t n_atoms_;
    torch::Tensor potential_energy_;
    torch::Tensor box_size_;
};

#endif