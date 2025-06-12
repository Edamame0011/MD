#ifndef LOAD_FILE
#define LOAD_FILE

#include "Atoms.hpp"
#include <vector>
#include <array>

namespace load_file{
    void load_xyz_structures(std::string data_path, std::vector<Atoms>& structures, bool is_ext = false, torch::Device device = torch::kCPU);   //構造データのロード
    void load_xyz_atoms(std::string data_path, Atoms& atoms, bool is_ext = false, torch::Device device = torch::kCPU);                          //単一構造のロード
}

#endif