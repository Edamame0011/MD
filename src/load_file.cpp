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
#include <stdexcept>

//補助用関数
namespace {
    //文字列からLatticeを見つける
    std::string find_lattice(std::string input) {
        //開始位置のキーワード
        std::string start_tag = "Lattice=\"";

        //開始位置
        std::size_t start_position = input.find(start_tag);

        if(start_position != std::string::npos){
            start_position = start_position + start_tag.length();

            //開始位置から次の"を探す
            std::size_t end_position = input.find('"', start_position);

            if(end_position != std::string::npos){
                //抜き出す部分の長さ
                std::size_t length = end_position - start_position;

                //文字列の抜き出し
                std::string result = input.substr(start_position, length);

                return result;
            }

            else{ throw std::runtime_error("終了のダブルクオーテーションが見つかりません。"); }
        }

        else{ throw std::runtime_error("ファイルにLatticeデータが含まれていません。"); }
    }
}

//複数構造の読み込み
void load_file::load_xyz_structures(std::string data_path, std::vector<Atoms>& structures, torch::Device device){
    std::vector<Atom> atoms_vector;
    torch::Tensor box_size;

    std::ifstream file(data_path);
    
    if(!file.is_open()){
        throw std::runtime_error("構造ファイルを開けません。");
    }
    
    std::string line;
    std::size_t num_atoms = 0;  //各構造の原子数
    std::size_t num_structures = 0; //構造の数

    int j = 0;  //一つの構造についての行数を表す変数

    while(std::getline(file, line)){
        //空白の行をスキップ
        if (line.empty() || line.find_first_not_of(" \t\n\v\f\r") == std::string::npos) {
            continue;
        }
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

                //ストリームから原子の種類、座標、力を取り出し、代入
                iss >> atom_type >> position_arr[0] >> position_arr[1] >> position_arr[2] >> force_arr[0] >> force_arr[1] >> force_arr[2];

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
                std::array<RealType, 3> lattice_x, lattice_y, lattice_z;
                //コメントからLatticeの部分を抜き出す
                std::string lattice = find_lattice(line);
                //文字列をストリームに変換
                std::istringstream iss(lattice);
                iss >> lattice_x[0] >> lattice_x[1] >> lattice_x[2] >> 
                       lattice_y[0] >> lattice_y[1] >> lattice_y[2] >>
                       lattice_z[0] >> lattice_z[1] >> lattice_z[2];
                //とりあえず正方格子を想定
                box_size = torch::tensor(lattice_x[0], torch::TensorOptions().device(device).dtype(kRealType));         
            }
            j ++;
        }
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

//extxyzフォーマットでない時
void load_file::load_xyz_structures(std::string data_path, std::vector<Atoms>& structures, float Lbox, torch::Device device){
    std::vector<Atom> atoms_vector;
    torch::Tensor box_size;

    std::ifstream file(data_path);
    
    if(!file.is_open()){
        throw std::runtime_error("構造ファイルを開けません。");
    }
    
    std::string line;
    std::size_t num_atoms = 0;  //各構造の原子数
    std::size_t num_structures = 0; //構造の数

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

                //文字列をストリームに変換
                std::istringstream iss(line);

                //ストリームから原子の種類、座標を取り出し、代入
                iss >> atom_type >> position_arr[0] >> position_arr[1] >> position_arr[2];

                //torch::Tensorに変換
                torch::Tensor position = torch::from_blob(position_arr.data(), {3}, kRealType).clone();

                //原子にセット
                Atom a;
                a.set_type(atom_type);
                a.set_position(position);
                atoms_vector.push_back(a);
            }

            else{
                //引数からbox_sizeを初期化
                box_size = torch::tensor(Lbox, torch::TensorOptions().device(device).dtype(kRealType));         
            }
            j ++;
        }
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
void load_file::load_xyz_atoms(std::string data_path, Atoms& atoms, torch::Device device){
    std::vector<Atom> atoms_vec;
    std::ifstream file(data_path);
    
    if(!file.is_open()){
        throw std::runtime_error("構造ファイルを開けません。");
    }
    
    std::string line;
    
    //1行目は原子数
    std::getline(file, line);

    //2行目はコメント行
    std::getline(file, line);

    std::array<RealType, 3> lattice_x, lattice_y, lattice_z;
    //latticeの部分を読み込む
    //コメントからlatticeの部分を抜き出す
    std::string lattice = find_lattice(line);
    //文字列をストリームに変換
    std::istringstream iss(lattice);
    iss >> lattice_x[0] >> lattice_x[1] >> lattice_x[2] >> 
           lattice_y[0] >> lattice_y[1] >> lattice_y[2] >>
           lattice_z[0] >> lattice_z[1] >> lattice_z[2];
    //とりあえず正方格子を想定
    torch::Tensor box_size = torch::tensor(lattice_x[0], torch::TensorOptions().device(device).dtype(kRealType));         

    //3行目以降
    while(std::getline(file, line)){
        //原子の情報を保持する変数
        Atom a;
        std::string atom_type;
        std::array<RealType, 3> position_arr;
        std::array<RealType, 3> force_arr;

        //文字列をストリームに変換
        std::istringstream iss(line);
        
        iss >> atom_type >> position_arr[0] >> position_arr[1] >> position_arr[2] >> force_arr[0] >> force_arr[1] >> force_arr[2];

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

    //出力
    std::cout << "単一構造を読み込みました。\n原子数：" << atoms.size().item<IntType>() << std::endl 
              << "ボックスのサイズ：" << atoms.box_size().item<RealType>() << std::endl; 
}

//単一構造の読み込み
void load_file::load_xyz_atoms(std::string data_path, Atoms& atoms, float Lbox, torch::Device device){
    std::vector<Atom> atoms_vec;
    std::ifstream file(data_path);
    
    if(!file.is_open()){
        throw std::runtime_error("構造ファイルを開けません。");
    }
    
    std::string line;
    
    //1行目は原子数
    std::getline(file, line);

    //2行目はコメント行
    std::getline(file, line);

    //引数から初期化
    torch::Tensor box_size = torch::tensor(Lbox, torch::TensorOptions().device(device).dtype(kRealType));         

    //3行目以降
    while(std::getline(file, line)){
        //原子の情報を保持する変数
        Atom a;
        std::string atom_type;
        std::array<RealType, 3> position_arr;

        //文字列をストリームに変換
        std::istringstream iss(line);
        
        iss >> atom_type >> position_arr[0] >> position_arr[1] >> position_arr[2];

        //torch::Tensorに変換
        torch::Tensor position = torch::from_blob(position_arr.data(), {3}, kRealType).clone();

        //原子にセット
        a.set_type(atom_type);
        a.set_position(position);

        atoms_vec.push_back(a);
    }

    atoms = Atoms(atoms_vec, device);
    atoms.set_box_size(box_size);
}