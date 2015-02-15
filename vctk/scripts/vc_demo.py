# coding: utf-8

"""
VCの設計にのっとったデモコード。

    設計：
    1. F0, スペクトル包絡、非周期成分、好きなように特徴抽出器を設定できる
    2. F0, スペクトル包絡、非周期成分、好きなように変換器を設定できる
    3, 分析・合成エンジンを切り替えられる

この例では、GMMベースのframe-by-frame 特徴量変換を行います

clb -> slt です

"""

from vctk import VoiceConverter
from vctk.backend import WORLD
from vctk.parameterization import MelCepstrumParameterizer
from vctk.parameterization import LogarithmicParameterizer
from vctk.parameterization import TransparentParameterizer
from vctk.conversion import GMMMap, Multiplier

import numpy as np
from sklearn.externals import joblib
from scipy.io import wavfile

from pylab import subplot, plot, show

if __name__ == "__main__":

    fs, x = wavfile.read("clb_a28.wav")
    x = np.array(x, dtype=np.float)

    # 分析・合成エンジンは、設計的には分けて使えるけど、今のところWORLDだけ
    engine = WORLD(period=5.0, fs=fs)

    # 学習済みのモデルを読み込み（sklearn.mixture.GMM）
    # まだデータの読み込みやパラレルデータの作成とか書いてないので、
    # https://github.com/r9y9/stav を使った
    # 学習条件は、clb_and_slt.yml に置いておいた
    gmm = joblib.load("gmm_clb_and_slt.pkl")

    # この例だと、
    # 1. F0を線形倍する（対数スケールで行いたい場合は、
    #    LogarithmicParameterizer使えばOK
    # 2. スペクトル包絡をGMMに基づいて変換する
    # 3. 非周期性指標はターゲット音声のをそのまま使う
    vc = VoiceConverter(
        # TrasparentParameterizerは、何もしない
        f0_parameterizer=TransparentParameterizer(),
        f0_converter=Multiplier(coef=1.2),
        # メルケプストラムにしてからGMM特徴量変換に渡す
        # paramterizerは、特徴量からスペクトル包絡に戻す役割も持つ
        spectrum_envelope_parameterizer=MelCepstrumParameterizer(
            order=40, alpha=0.41, fftlen=1024),
        # GMMベースの変換器を設定する
        spectrum_envelope_converter=GMMMap(gmm=gmm),
        aperiodicity_parameterizer=None,
        aperiodicity_converter=None,
        # 分析、合成エンジン
        analyzer=engine,
        synthesizer=engine
    )

    # これだけ実行すればおｋ
    vc.analyze(x)
    vc.convert()
    y = vc.synthesis()

    # デバッグ用に可視化
    time_axis = np.linspace(0, len(x) / float(fs), len(x))
    subplot(2, 1, 1)
    plot(time_axis, x)
    subplot(2, 1, 2)
    plot(time_axis, y)
    show()

    wavfile.write("slt_a28.wav", fs, np.array(y, dtype=np.int16))
