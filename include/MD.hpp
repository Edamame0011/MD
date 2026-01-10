/**
* @file MD.hpp
* @brief MDクラス
*/

#ifndef MD_HPP
#define MD_HPP

#include "Atoms.hpp"
#include "NeighbourList.hpp"
#include "config.h"
#include "NoseHooverThermostat.hpp"
#include "BussiThermostat.hpp"

#include <torch/script.h>
#include <torch/torch.h>

#include <optional>

class MD{
    public:
        //コンストラクタ
        MD(torch::Tensor dt, torch::Tensor cutoff, torch::Tensor margin, std::string data_path, std::string model_path, torch::Device device = torch::kCPU); 
        MD(RealType dt, RealType cutoff, RealType margin, std::string data_path, std::string model_path, torch::Device device = torch::kCPU); 
        MD(RealType dt, RealType cutoff, RealType margin, const Atoms& atoms, torch::Device device = torch::kCPU); 

        //シミュレーション
        /**
         * @brief NVEシミュレーションの実行
         * @param[in] tsim シミュレーション時間 (fs)
         * @param[in] temp 初期温度
         * @param[in] step 何ステップごとに出力するか
         * @param[in] is_save 各ステップごとにtrajectoryを保存するか
         */
        void NVE(const RealType tsim, const RealType temp, const IntType step, const bool is_save = false);
        /**
         * @brief NVEシミュレーションの実行
         * @param[in] tsim シミュレーション時間 (fs)
         * @param[in] temp 初期温度
         * @param[in] log その他の保存方法（現在はlogスケールのみ）
         * @param[in] is_save 各ステップごとにtrajectoryを保存するか
         */
        void NVE(const RealType tsim, const RealType temp, const std::string log, const bool is_save = false);

        //一定温度のシミュレーション
        /**
         * @brief NVTシミュレーションの実行
         * 
         * 一定温度のNVTシミュレーション
         * 
         * @param[in] tsim シミュレーション時間 (fs)
         * @param[in] Thermostat 熱浴
         * @param[in] step 何ステップごとに出力するか
         * @param[in] is_save 各ステップごとにtrajectoryを保存するか
         * 
         * @note 熱浴に、あらかじめ目標温度を設定しておいてください。
         */
        template <typename ThermostatType>
        void NVT(const RealType tsim, ThermostatType& Thermostat, const IntType step, const bool is_save = false);            
        /**
         * @brief NVTシミュレーションの実行
         * 
         * 一定温度のNVTシミュレーション
         * 
         * @param[in] tsim シミュレーション時間 (fs)
         * @param[in] Thermostat 熱浴
         * @param[in] log その他の保存方法（現在はlogスケールのみ）
         * @param[in] is_save 各ステップごとにtrajectoryを保存するか
         * @param[in] N_per_decade 1 decade あたりのサンプル数 N
         * @param[in] M_boost 各点で連続保存するステップ数 M
         * @param[in] interval_boost 連続保存する際のステップ間隔
         * 
         * @note 熱浴に、あらかじめ目標温度を設定しておいてください。
         * 
         */
        template <typename ThermostatType>
        void NVT(const RealType tsim, ThermostatType& Thermostat, const std::string log, const bool is_save = false,const IntType N_per_decade = 5, const IntType M_boost = 10, const IntType interval_boost = 10);        

        //温度変化をさせるシミュレーション
        /**
         * @brief NVTシミュレーションの実行
         * 
         * 温度を変化させるシミュレーション
         * 
         * @param[in] cooling_rate 冷却速度 (K/fs)
         * @param[in] Thermostat 熱浴
         * @param[in] targ_temp 目標温度 (K)
         * @param[in] step 何ステップごとに出力するか
         * @param[in] is_save 各ステップごとにtrajectoryを保存するか
         * 
         * @note 熱浴に、あらかじめ初期温度を設定しておいてください。
         * 
         */
        template <typename ThermostatType>
        void NVT_anneal(const RealType cooling_rate, ThermostatType& Thermostat, const RealType targ_temp, const IntType step, const bool is_save = false);
        /**
         * @brief NVTシミュレーションの実行
         * 
         * 温度を変化させるシミュレーション
         * 
         * @param[in] cooling_rate 冷却速度 (K/fs)
         * @param[in] Thermostat 熱浴
         * @param[in] targ_temp 目標温度 (K)
         * @param[in] log その他の保存方法（現在はlogスケールのみ）
         * @param[in] is_save 各ステップごとにtrajectoryを保存するか
         * @param[in] N_per_decade 1 decade あたりのサンプル数 N
         * @param[in] M_boost 各点で連続保存するステップ数 M
         * @param[in] interval_boost 連続保存する際のステップ間隔
         * 
         * @note 熱浴に、あらかじめ初期温度を設定しておいてください。
         * 
         */
        template <typename ThermostatType>
        void NVT_anneal(const RealType cooling_rate, ThermostatType& Thermostat, const RealType targ_temp, const std::string log, const bool is_save = false,const IntType N_per_decade = 5, const IntType M_boost = 10, const IntType interval_boost = 10);

        //原子の保存
        /**
         * @brief 現在の系を保存
         * @param[in] save_path 保存するパス
         */
        void save_atoms(const std::string& save_path);
        /**
         * @brief 現在の系を保存
         * 
         * pbcをアンラップして保存
         * 
         * @param[in] save_path 保存するパス
         */
        void save_unwrapped_atoms(const std::string& save_path);

        /**
         * @brief 温度をもとに、原子の速度を初期化
         * 
         * @param[in] initial_temp 温度 (K)
         */
        void init_temp(const RealType initial_temp);                     //原子の速度（温度）の初期化

        /**
         * @brief ステップ数を0に戻す
         */
        void reset_step();
        /**
         * @brief 原子が何個目のミラーにあるかを保存する配列をリセット
         */
        void reset_box();
        /**
         * @brief trajectoryファイルの保存先を変更
         */
        void set_traj_path(const std::string& path);

        /**
         * @brief 系の読み込み
         */
        void load_atoms(const std::string& path);

        /**
         * @brief 現在の運動温度を取得
         */
        RealType kinetic_temperature() const {
            return atoms_.temperature().item<RealType>();
        }

        //テスト用
        void NVE_LJ(const RealType tsim, const RealType temp, const IntType step, const bool is_save = false, const std::string output_path = "./data/saved_structure.xyz");

        template <typename ThermostatType>
        void NVT_LJ(const RealType tsim, ThermostatType& Thermostat, const std::string log, const bool is_save = false, const std::string output_path = "./data/saved_structure.xyz");        

        template <typename ThermostatType>
        void NVT_LJ(const RealType tsim, ThermostatType& Thermostat, const IntType step, const bool is_save = false, const std::string output_path = "./data/saved_structure.xyz");        


    private:
        //その他（補助用関数）
        /**
         * @brief 経過時間・運動エネルギー・ポテンシャルエネルギー・全エネルギー・温度を出力
         */
        void print_energies();                                          //結果の出力

        /**
         * @brief NVEシミュレーションを1ステップ行う
         */
        void step();                                  //1ステップ
        /**
         * @brief NVTシミュレーションを1ステップ行う
         * 
         * @param[in] Thermostat nose-hoover熱浴
         */
        void step(NoseHooverThermostat& Thermostat);
        /**
         * @brief NVTシミュレーションを1ステップ行う
         * 
         * @param[in] Thermostat bussir熱浴
         */
        void step(BussiThermostat& Thermostat);

        /**
         * @brief NVEシミュレーションのメインループ
         * 
         * @param[in] tsim シミュレーション時間 (fs)
         * @param[in] temp 初期温度
         * @param[in] output_action 出力関数
         */
        template <typename OutputAction>
        void NVE_loop(const RealType tsim, const RealType temp, OutputAction output_action);

        /**
         * @brief NVTシミュレーションのメインループ
         * 
         * @param[in] tsim シミュレーション時間 (fs)
         * @param[in] Thermostat 熱浴
         * @param[in] output_action 出力関数
         */
        template <typename OutputAction, typename ThermostatType>
        void NVT_loop(const RealType tsim, ThermostatType& Thermostat, OutputAction output_action);

        /**
         * @brief NVTシミュレーションのメインループ
         * 
         * 温度を変化させながらシミュレーション
         * 
         * @param[in] cooling_rate 冷却速度 (K/fs)
         * @param[in] Thermostat 熱浴
         * @param[in] output_action 出力関数
         */
        template <typename OutputAction, typename ThermostatType>
        void NVT_anneal_loop(const RealType cooling_rate, ThermostatType& Thermostat, const RealType targ_temp, OutputAction output_action);

        //テスト用
        void step_LJ(torch::Tensor& box);                                  //1ステップ
        void step_LJ(torch::Tensor& box, NoseHooverThermostat& Thermostat);
        void step_LJ(torch::Tensor& box, BussiThermostat& Thermostat);

        template <typename OutputAction>
        void NVE_loop_LJ(const RealType tsim, const RealType temp, OutputAction output_action);

        template <typename OutputAction, typename ThermostatType>
        void NVT_loop_LJ(const RealType tsim, ThermostatType& Thermostat, OutputAction output_action);

        template <typename OutputAction, typename ThermostatType>
        void NVT_anneal_loop_LJ(const RealType cooling_rate, ThermostatType& Thermostat, const RealType targ_temp, OutputAction output_action);

        //シミュレーション用
        IntType t_;                                                     //現在のステップ数
        RealType temp_;                                                 //現在の温度
        torch::Tensor dt_;                                              //時間刻み幅
        RealType dt_real_;
        torch::Tensor Lbox_;                                            //シミュレーションセルのサイズ
        torch::Tensor Linv_;                                            //セルのサイズの逆数
        NeighbourList NL_;                                              //隣接リスト

        torch::Tensor box_;                                             //周期境界条件のもとで、何個目の箱のミラーに位置しているのかを保存する変数 (N, 3)
        std::string traj_path_;                                         //trajectoryを保存するパス

        //MLP用変数
        torch::jit::script::Module module_;                              //モデルを格納する変数

        //系
        Atoms atoms_;                                                    //原子
        torch::Tensor num_atoms_;                                        //原子数

        //シミュレーションデバイス
        torch::Device device_;

        //定数
        torch::Tensor boltzmann_constant_;
        torch::Tensor conversion_factor_;
};

#include "MD.tpp"

#endif