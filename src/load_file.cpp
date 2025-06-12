#include "load_file.hpp"
#include "Atoms.hpp"
#include "config.h"

#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cctype>
#include <algorithm>
#include <map>

//複数構造の読み込み
void load_file::load_xyz_structures(std::string data_path, std::vector<Atoms>& structures, bool is_ext, torch::Device device){
    std::vector<Atom> atoms_vector;
    torch::Tensor box_size;

    std::ifstream file(data_path);
    
    if(!file.is_open()){
        std::cerr << "ファイルを開けませんでした：" << data_path << std::endl;
        throw;
    }
    
    std::string line;
    std::size_t num_atoms = 0;  //各構造の原子数
    std::size_t num_structures = 0; //構造の数

    int i = 0;  //ループ数を表す変数
    int j = 0;  //一つの構造についての行数を表す変数

    while(std::getline(file, line)){
        //lineが数値かどうかを判定
        if(std::all_of(line.cbegin(), line.cend(), isdigit)){
            num_atoms = std::stoi(line);

            j = 0;

            //structuresにatomsを追加
            if(!atoms_vector.empty()){
                Atoms atoms = Atoms(atoms_vector, device);
                atoms.set_box_size(box_size);
                structures.push_back(atoms);
                num_structures ++;
                atoms_vector.clear();
            }
        }
        else{
            if(j != 0){
                //原子の情報を保持する変数
                std::string atom_type;
                std::array<RealType, 3> position_arr;
                std::array<RealType, 3> force_arr;

                //文字列をストリームに変換
                std::istringstream iss(line);

                //ストリームから原子の種類、座標を取り出し、代入
                //format="extxyz"の時は、力も保存されている。
                if(is_ext){
                    iss >> atom_type >> position_arr[0] >> position_arr[1] >> position_arr[2] >> force_arr[0] >> force_arr[1] >> force_arr[2];
                }
                else{
                    iss >> atom_type >> position_arr[0] >> position_arr[1] >> position_arr[2];
                }

                //torch::Tensorに変換
                torch::Tensor position = torch::from_blob(position_arr.data(), {3}, kRealType).clone();
                torch::Tensor force = torch::from_blob(force_arr.data(), {3}, kRealType).clone();

                //原子にセット
                Atom a;
                a.set_type(atom_type);
                a.set_position(position);
                a.set_force(force);
                atoms_vector.push_back(a);
            }

            else{
                //とりあえず定数で初期化
                box_size = torch::tensor(10.585286, torch::TensorOptions().device(device).dtype(kRealType));
            }
            j ++;
        }
        i ++;
    }

    if(!atoms_vector.empty()){
        Atoms atoms = Atoms(atoms_vector, device);
        atoms.set_box_size(box_size);
        structures.push_back(atoms);
        num_structures++;
        atoms_vector.clear();
    }

    std::cout << "複数の構造をロードしました。" << std::endl;
    std::cout << "構造の数：" << num_structures << std::endl;
}

//単一構造の読み込み
void load_file::load_xyz_atoms(std::string data_path, Atoms& atoms, bool is_ext, torch::Device device){
    std::vector<Atom> atoms_vec;
    std::ifstream file(data_path);
    
    if(!file.is_open()){
        std::cerr << "ファイルを開けませんでした：" << data_path << std::endl;
        throw;
    }
    
    std::string line;
    
    //1行目は原子数
    std::getline(file, line);

    //2行目はコメント行
    std::getline(file, line);
    //とりあえず10で初期化している。本当はファイルから読み込むべき。
    torch::Tensor box_size = torch::tensor(10.0, torch::TensorOptions().dtype(kRealType).device(device));  

    //3行目以降
    while(std::getline(file, line)){
        //原子の情報を保持する変数
        Atom a;
        std::string atom_type;
        std::array<RealType, 3> position_arr;
        std::array<RealType, 3> force_arr;

        //文字列をストリームに変換
        std::istringstream iss(line);
        
        if(is_ext){
            iss >> atom_type >> position_arr[0] >> position_arr[1] >> position_arr[2] >> force_arr[0] >> force_arr[1] >> force_arr[2];
        }
        else{
            //ストリームから読み込み
            iss >> atom_type >> position_arr[0] >> position_arr[1] >> position_arr[2];
        }

        //torch::Tensorに変換
        torch::Tensor position = torch::from_blob(position_arr.data(), {3}, kRealType).clone();
        torch::Tensor force = torch::from_blob(force_arr.data(), {3}, kRealType).clone();

        //原子にセット
        a.set_type(atom_type);
        a.set_position(position);
        a.set_force(force);

        atoms_vec.push_back(a);
    }

    atoms = Atoms(atoms_vec, device);
    atoms.set_box_size(box_size);
}