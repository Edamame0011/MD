#ifndef CONFIG_H
#define CONFIG_H

#include <torch/torch.h>
#include <torch/script.h>

//原子種類と原子番号を関連づけるmap
inline std::map<std::string, int> atom_number_map = {
        {"H",  1},
        {"He", 2},
        {"Li", 3},
        {"Be", 4},
        {"B",  5},
        {"C",  6},
        {"N",  7},
        {"O",  8},
        {"F",  9},
        {"Ne", 10},
        {"Na", 11},
        {"Mg", 12},
        {"Al", 13},
        {"Si", 14},
        {"P",  15},
        {"S",  16},
        {"Cl", 17},
        {"Ar", 18},
        {"K",  19},
        {"Ca", 20}
    };

//原子種類と原子質量を関連づけるmap
inline std::map<std::string, double> atom_mass_map = {
        {"H",   1.0080},
        {"He",  4.0026},
        {"Li",  6.94},
        {"Be",  9.0122},
        {"B",   10.81},
        {"C",   12.011},
        {"N",   14.007},
        {"O",   15.999},
        {"F",   18.998},
        {"Ne",  20.180},
        {"Na",  22.990},
        {"Mg",  24.305},
        {"Al",  26.982},
        {"Si",  28.0855},
        {"P",   30.974},
        {"S",   32.06},
        {"Cl",  35.45},
        {"Ar",  39.95},
        {"K",   39.098},
        {"Ca",  40.078}
    };

//精度の設定
using RealType = float;
constexpr torch::ScalarType kRealType = torch::kFloat32;

using IntType = int;
constexpr torch::ScalarType kIntType = torch::kInt64;

//定数
const RealType unit_fs = 1e-15;
//ボルツマン定数 (eV / K)
const torch::Tensor boltzmann_constant = torch::tensor(8.617333262145e-5, torch::TensorOptions().dtype(kRealType));
//変換係数 (ev / amu) -> ((Å / fs) ^ 2)
const torch::Tensor conversion_factor = torch::tensor(0.964855e-2, torch::TensorOptions().dtype(kRealType));

#endif

/*
単位換算
//(eV / u) -> (Å / (fs^2))
(eV / u) = (e * J / (1.66054 * 10^-27 kg))
         = (e * 10^27 * 1.66054^-1 kg m^2 / (kg * s^2))
         = (e * 10^27 * 1.66054^-1 m^2 / (s^2))
         = (e * 10^27 * 1.66054^-1 * 10^20 Å^2 / ((10^15)^2 fs^2))
         = (e * 10^27 * 1.66054^-1 * 10^20 * 10^-30 Å^2 / (fs^2))
         = (0.964855 * 10^-2 Å^2 / (fs^2))

//(Å^2 u / (fs^2)) -> (eV)
(Å^2 u / (fs^2)) = (10^-20 * 1.66054 * 10^-27 m^2 kg / ((10^-15)^2 s^2))
                 = (10^30 * 10^-20 * 10^-27 * 1.66054 m^2 kg / s^2)
                 = (10^-17 * 1.66054 J)
                 = (1.66054 * 10^-17 * e^-1 eV)
                 = ((0.964855 * 10^-2)^-1 eV)

//(eV / (Å・u)) -> (Å / (fs^2))
(eV/(Å・u)) = (e * J / (10^-10 m * 1.66054 * 10^-27 kg))
            = (e * 10^27 * 10^10 * 1.66054^-1 kg m^2 / (m * kg * s^2))
            = (e * 10^37 * 1.66054^-1 m / s^2)
            = (e * 10^37 * 1.66054^-1 * 10^10 Å / ((10^15)^2 fs^2))
            = (e * 10^37 * 1.66054^-1 * 10^10 * 10^-30 Å / (fs^2))
            = (e * 10^17 * 1.66054^-1 Å / (fs^2))
            = (1.60218 * 10^-19 * 10^17 * 1.66054^-1 Å / (fs^2))
            = (0.964855 * 10^-2 Å / (fs^2))
*/