#include "Atom.hpp"
#include "config.h"

#include <iostream>
#include <map>

//コンストラクタ
Atom::Atom(std::string type, torch::Tensor position, torch::Tensor velocity, torch::Tensor force, torch::Device device)
    : type_(std::move(type)), device_(device)
    {   
        //deviceに移動させてから初期化
        position_ = position.to(device_);
        velocity_ = velocity.to(device_);
        force_ = force.to(device_);

        //形状のチェック
        TORCH_CHECK(position.sizes() == torch::IntArrayRef{3}, "Position tensor must have shape {3}");
        TORCH_CHECK(velocity.sizes() == torch::IntArrayRef{3}, "Velocity tensor must have shape {3}");
        TORCH_CHECK(force.sizes() == torch::IntArrayRef{3}, "Force tensor must have shape {3}");

        //原子番号と質量の初期化
        auto options = torch::TensorOptions().device(device_);
        atomic_number_ = torch::tensor(atom_number_map[type_], options.dtype(torch::kInt64));
        mass_ = torch::tensor(atom_mass_map[type_], options.dtype(kRealType));

        //使用する定数のデバイスを移動させておく。
        conversion_factor.to(device);
    }

Atom::Atom(std::string type, std::array<double, 3>& position, std::array<double, 3>& velocity, std::array<double, 3>& force, torch::Device device)
     : Atom(std::move(type), 
            torch::from_blob(const_cast<double*>(position.data()), {3}, kRealType).clone(), 
            torch::from_blob(const_cast<double*>(velocity.data()), {3}, kRealType).clone(), 
            torch::from_blob(const_cast<double*>(force.data()), {3}, kRealType).clone(), 
            device){}

Atom::Atom()
     : Atom(
        std::move("H"), 
        torch::zeros(3, kRealType), 
        torch::zeros(3, kRealType), 
        torch::zeros(3, kRealType), 
        torch::kCPU
     ){}

//ゲッタ
const torch::Tensor Atom::mass_inv() const {
    if((mass_ != 0).item<bool>()){
        return 1.0 / mass_;
    }
    else{
        throw std::runtime_error("質量がゼロです。");
    }
}

//セッタ
void Atom::set_type(std::string& type){
    type_ = type;

    //原子番号と質量の初期化
    auto options = torch::TensorOptions().device(device_);
    atomic_number_ = torch::tensor(atom_number_map[type_], options.dtype(torch::kInt64));
    mass_ = torch::tensor(atom_mass_map[type_], options.dtype(kRealType));
}

void Atom::set_position(torch::Tensor& position){
    position_ = position;

    //形状のチェック
    TORCH_CHECK(position.sizes() == torch::IntArrayRef{3}, "Position tensor must have shape {3}");
}

void Atom::set_velocity(torch::Tensor& velocity){
    velocity_ = velocity;

    //形状のチェック
    TORCH_CHECK(velocity.sizes() == torch::IntArrayRef{3}, "Position tensor must have shape {3}");
}

void Atom::set_force(torch::Tensor& force){
    force_ = force;

    //形状のチェック
    TORCH_CHECK(force.sizes() == torch::IntArrayRef{3}, "Position tensor must have shape {3}");
}

void Atom::set_position(std::array<double, 3> position){
    torch::Tensor position_tensor = torch::from_blob(const_cast<double*>(position.data()), {3}, kRealType).clone();
    set_position(position_tensor);
}

void Atom::set_velocity(std::array<double, 3> velocity){
    torch::Tensor velocity_tensor = torch::from_blob(const_cast<double*>(velocity.data()), {3}, kRealType).clone();
    set_velocity(velocity_tensor);
}

void Atom::set_force(std::array<double, 3> force){
    torch::Tensor force_tensor = torch::from_blob(const_cast<double*>(force.data()), {3}, kRealType).clone();
    set_force(force_tensor);
}

void Atom::to(torch::Device device){
    device_ = device;
    atomic_number_ = atomic_number_.to(device);
    mass_ = mass_.to(device);
    position_ = position_.to(device);
    velocity_ = velocity_.to(device);
    force_ = force_.to(device);
}

torch::Tensor Atom::kinetic_energy(){
    torch::Tensor kinetic_energy = 0.5 * mass_ * ( velocity_[0] * velocity_[0] + 
                                                   velocity_[1] * velocity_[1] + 
                                                   velocity_[2] * velocity_[2]);
    //1/2 mv^2 (u * (Å / fs) ^ 2) -> (eV)
    return kinetic_energy / conversion_factor;
}