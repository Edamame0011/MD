#ifndef ATOM_HPP
#define ATOM_HPP

#include <string>
#include <array>
#include <iostream>
#include <map>

#include <torch/script.h>
#include <torch/torch.h>

class Atom{
    public:
        //コンストラクタ
        Atom();
        Atom(std::string type, torch::Tensor position, torch::Tensor velocity, torch::Tensor force, torch::Device device);
        Atom(std::string type, std::array<double, 3>& position, std::array<double, 3>& velocity, std::array<double, 3>& force, torch::Device device);

        //ゲッタ
        const std::string& type() const { return type_; }
        const torch::Tensor& atomic_number() const { return atomic_number_; }
        const torch::Tensor& mass() const { return mass_; }
        const torch::Tensor mass_inv() const;
        const torch::Tensor& position() const { return position_; }
        const torch::Tensor& velocity() const { return velocity_; }
        const torch::Tensor& force() const { return force_; }
        const torch::Device& device() const { return device_; }

        //セッタ
        void set_type(std::string& type);
        void set_position(torch::Tensor& position);
        void set_velocity(torch::Tensor& velocity);
        void set_force(torch::Tensor& force);

        void set_position(std::array<double, 3> position);
        void set_velocity(std::array<double, 3> velocity);
        void set_force(std::array<double, 3> force);

        void to(torch::Device device);

        //その他
        torch::Tensor kinetic_energy();    //運動エネルギーを計算
    private:
        //原子種類
        std::string type_;
        torch::Tensor atomic_number_;
        torch::Tensor mass_;

        //プロパティ
        torch::Tensor position_;
        torch::Tensor velocity_;
        torch::Tensor force_;

        //計算デバイス
        torch::Device device_;
};

#endif