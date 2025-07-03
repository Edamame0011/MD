#ifndef LOAD_FILE
#define LOAD_FILE

#include "Atoms.hpp"
#include <vector>
#include <array>

namespace xyz{
    //構造の読み込み
    void load_structures(std::string data_path, std::vector<Atoms>& structures, torch::Device device = torch::kCPU);                        //構造データのロード
    void load_structures(std::string data_path, std::vector<Atoms>& structures, float Lbox, torch::Device device = torch::kCPU);            //extxyzフォーマットじゃない時
    void load_atoms(std::string data_path, Atoms& atoms, torch::Device device = torch::kCPU);                                               //単一構造のロード
    void load_atoms(std::string data_path, Atoms& atoms, float Lbox, torch::Device device = torch::kCPU);                                   //extxyzフォーマットじゃない時

    //構造の保存
    void save_atoms(std::string data_path, Atoms atoms);
}

#endif