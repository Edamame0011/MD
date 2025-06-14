#ifndef LOAD_FILE
#define LOAD_FILE

#include "Atoms.hpp"
#include <vector>
#include <array>

namespace load_file{
    void load_xyz_structures(std::string data_path, std::vector<Atoms>& structures, torch::Device device = torch::kCPU);                        //構造データのロード
    void load_xyz_structures(std::string data_path, std::vector<Atoms>& structures, float Lbox, torch::Device device = torch::kCPU);            //extxyzフォーマットじゃない時
    void load_xyz_atoms(std::string data_path, Atoms& atoms, torch::Device device = torch::kCPU);                                               //単一構造のロード
    void load_xyz_atoms(std::string data_path, Atoms& atoms, float Lbox, torch::Device device = torch::kCPU);                                   //extxyzフォーマットじゃない時                  
}

#endif