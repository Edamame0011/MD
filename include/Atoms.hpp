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
    const torch::Tensor& size() const { return n_atoms_; }
    const std::vector<std::string>& types() const { return types_; }

    //セッタ
    void set_positions(const torch::Tensor& positions);
    void set_velocities(const torch::Tensor& velocities);
    void set_forces(const torch::Tensor& forces);
    void set_box_size(const torch::Tensor& box_size);
    void set_potential_energy(const torch::Tensor& potential_energy);

    //物理量の計算
    torch::Tensor kinetic_energy() const;
    torch::Tensor potential_energy() const { return potential_energy_; }

    //その他
    void positions_update(const torch::Tensor dt, torch::Tensor& box);
    void velocities_update(const torch::Tensor dt);
    void apply_pbc(); //周期境界条件の補正
    void apply_pbc(torch::Tensor& box);
    
private:
    //計算デバイス
    torch::Device device_;

    //各原子のデータ
    torch::Tensor positions_;   //(num_atoms, 3)
    torch::Tensor velocities_;  //(num_atoms, 3)
    torch::Tensor forces_;      //(num_atoms, 3)
    torch::Tensor masses_;      //(num_atoms, )
    torch::Tensor atomic_numbers_;  //(num_atoms, )
    std::vector<std::string> types_;

    //系のデータ
    torch::Tensor n_atoms_;
    torch::Tensor potential_energy_;
    torch::Tensor box_size_;
};

#endif