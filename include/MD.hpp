#ifndef MD_HPP
#define MD_HPP

#include "Atoms.hpp"
#include "NeighbourList.hpp"

#include <random>
#include <torch/script.h>
#include <torch/torch.h>

class MD{
    public:
        //コンストラクタ
        MD(T dt0, T cutoff0, T margin0, std::string data_path, std::string model_path);                         //xyzファイルから初期構造をロード
        MD(T dt0, T cutoff0, T margin0, std::size_t num_atoms, std::string atomType, T box_size, std::string model_path);   //初期構造をランダムに作成

        //初期化
        void init_vel_MB(const float float_targ, std::mt19937 &mt);    //原子の速度の初期化

        //シミュレーション
        void NVE(const T tsim);
        void NVE_LJ(const T tsim);
        void NVE_no_NL(const T tsim, const T cutoff);

    private:
        //隣接リスト
        void generate_NL();                                             //隣接リストの作成
        void NL_check();                                                //隣接リストの確認

        //系の更新
        void velocity_update();                                         //速度の更新
        void position_update();                                         //位置の更新

        //その他（補助用関数）
        void print_energies(T t);                                       //結果の出力
        void remove_drift();                                            //全体速度の除去
        
        void make_random(std::size_t num_atoms, std::string atomType, T box_size);  //ランダムな初期構造を作成

        //シミュレーション用
        T dt;                                                           //時間刻み幅
        T Lbox;
        T Linv;
        NeighbourList<T> neighbour_list;                                 //隣接リスト

        //MLP用変数
        torch::jit::script::Module module;                              //モデルを格納する変数

        //系
        Atoms<T> atoms;                                                 //原子
        std::size_t N;                                                  //原子数
};

#endif