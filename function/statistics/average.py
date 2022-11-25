import math
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing

import matplotlib.pyplot as plt


def spearman(data, target_1='', target_2='', ret=True):
    """
    ターゲットとするデータは順位である必要あり。
    2組の順位の間にどのような関係があるかを調べる。
    ------------------------------------------
    2つのデータセット間の関係の単調性のノンパラメトリック尺度
    ピアソンとは違い、２つのデータセットが正規分布している事を前提とはしない
    ------------------------------------------
    ｐ値は、相関のないシステムが、これらのデータセットから計算されたものと少なくとも同じ
    くらい極端なスピアマン相関を持つデータセットを生成する確率を大まかに示す。
    p値は完全に信頼できるわけではないが、500程度を超えるデータセットには妥当
    ------------------------------------------
    データにnanが含まれる場合の処理
        nan_policy= :
            propagate : (default) nanを返す
            raise     : エラーをスローする
            omit      : nanを無視
    ------------------------------------------
    相関係数=1-6{Σ(x-y)^2}/n{(n)^2-1}
    ------------------------------------------
    :param data: 元データ
    :param target_1: 順位ターゲット１
    :param target_2: 順位ターゲット２
    :param ret: True=> correlation & pvalue, False=>correlation
    :return: float 2-D ndarray
    """
    df1 = data[target_1]
    df2 = data[target_2]
    correlation, pvalue = stats.spearmanr(df1, df2)
    print(f'順位相関係数 : {correlation}')
    if ret is True:
        return correlation, pvalue
    else:
        return correlation


def dispersion_add(data, targetA, targetB):
    """
    N(μ1, σ1^2) + N(μ2, σ2^2) = N(μ1+μ2, σ1^2+σ1^2)
    N(2μ, 2σ^2) / 2 = N(μ, 2^2/2)
    (例)　A, Bグループ、　A＝N(165, (6)^2) B=(155, (5)^2)
         N(165-155, √6^2+5^2)
         mean = 165 - 155 = 10
         variance = √6^2 + 5^2 = 7.8
         平均値よりもさらに10以上差のある組み合わせは何%かを調べる
    :data: DataFrame
    :return:
    """
    dataA = data[targetA]
    dataB = data[targetB]
    #mean = 165 - 155
    #std = 7.8
    mean = dataA.mean() - dataB.mean()
    std = np.std(dataA, ddof=1) + np.std(dataB, ddof=1)
    Z = round(mean / std, 2)
    norm = round(stats.norm.cdf(x=Z) - 0.5, 4)  # 正規分布表参照方法
    per = 0.5 - norm
    print(f"mean : {mean}  < difference combination {round(per * 100)}%")


def unbased_varinace(data):
    """
    母集団の分散(母分散)を推定する。母集団からｎ個の標本を抜き取って母分散の推定値を求める式
    (n - μ)^2 / n
    が成立するが、母平均は一般にわからないのが普通であるので、母平均の代わりに標本平均を使って
    標本分散を計算することを考えるが標本分散は母分散の推定値とはならないという問題が起こる。
    (n-1)という自由度を使用して偏差平方和を割る事を不偏分散という。つまり以下の式
    (n - x_)^2 / (n - 1)
    これは　np.var(x, ddof=1)　で実行可能だが、後々の拡張として関数化しておく
    :return:
    """
    n = len(data)
    dds = deviation_sum_of_square(data)
    var = dds / n
    print(f'不偏分散 : {round(var, 2)}')
    return var


def deviation_sum_of_square(data):
    """
    偏差平方和
        データのばらつきの大きさを示す
        偏差平方和 = sum(data**2) - sum(data)**2 / len(data)
    :return:
    """
    data = pd.DataFrame(data)
    dss = round((np.sum(data ** 2) - np.sum(data) ** 2 / len(data))[0])
    print(f'偏差平方和：{dss}')
    return dss


def Reconciliation(X, U, V):
    """
    調和平均
    <平均時速計算>
    データの数をデータの逆数の和で割ったもの。
    速度の平均に使用 :
        |X＞方向をU、＜X|方向をV　とした時の平均時速
        2 / 1/U + 1/V = 1 / (1/U + 1/V) / 2 = 1 / (Uの逆数＋Vの逆数) / 2
    :X: 距離
    :U: |X>方向平均時速
    :V: <X|方向平均時速
    :return:
    """
    uh = X/U
    vh = X/V
    uv_time = uh + vh
    XX = X*2
    hm_h = XX / uv_time

    print(f'U方向所要時間 : {uh}, V方向所要時間 : {vh}, 往復合計 : {uv_time}')
    print(f'総移動距離 : {uv_time}')
    print(f'往復平均時速 : {hm_h}')


def geometric_average(df, target=""):
    """
    幾何平均
    幾何平均 = ｎ√データ積 = n√np.prod(n)
    (例):毎年の伸び率の平均何%なのかの計算に使用
        :算術平均ではデータ外れ値がある場合に平均が歪む。(外れ値検定参照)
    :param n:　Pandas-Columns
    :return:  geometric_average
    """
    ga = stats.gmean(df[target])
    print("Target : {}, 幾何平均 : {}%".format(target, round(ga, 1)))
    return ga


def diff_dataframe(df, target='', window=3, center=False):
    """
    Diff_PM : 全データ差分
    Rate_change : 変化率
    Moving_average : 移動平均
    :param df:　データフレーム
    :param target:　差分計算ターゲット
    :param window:　窓関数幅の指定、デフォルト３
    :param center:　デフォルトFalse
    :return:　df
    """
    df['Diff_PM'] = df[target].diff()
    df['Rate_change'] = df[target].pct_change()
    df['Moving_Average'] = df[target].rolling(window=window, center=center).mean()
    return df


def group_dataframe(df, timestamp='', target='', shift=4, window=4, center=True):
    """
    Growth rate from the previous period : 前期比伸び率(GRPD)
    Year-on-year growth rate　: 前年同期比伸び率(YoYGR)
    Average fourth-half movement　:　四半期移動平均(AFHM)
    四半期ごとのグループ化
    :param df: データフレーム
    :param timestamp: 四半期ごとのグループ化する時系列データの指定
    :return: df
    """
    target = str(target)
    timestamp = str(timestamp)
    df.set_index(pd.DatetimeIndex(df[timestamp]), inplace=True)
    df.set_index([df.index.quarter, df.index.year, df.index], inplace=True)
    df.index.names = ['quarter', 'year', 'date']
    df = df.mean(level=['quarter', 'year'])
    df['GRPD'] = df[target]
    df['YoYGR'] = (df[target] - df[target].shift(shift) / df[target])
    df['AFHM'] = df[target].rolling(window=window, center=center).mean()
    return df


def normalized_distribution_std(df, target="", normalize='sample', axis=0):
    """
    N(μ, σ) -> N(0, 1)
    正規分布の正規化
        (xi - μ)/σ
    母集団標準偏差
        σ^2 = 1/n Σn i=1 (xi - μ)^2
    標本標準偏差
        x_ = 1/n-1 Σn i=1 (xi - x_)^2
    default : stats.zscore(x, ddof=0)
    :return:
    """
    target = str(target)
    df = pd.DataFrame(df)
    if normalize == "population":
        df['zscore_population'] = stats.zscore(df[target], ddof=0, axis=0)
    elif normalize == "sample":
        df['zscore_sample'] = stats.zscore(df[target], ddof=1, axis=0)
    else:
        print('Normalization : sample (n-1) or population (n)')
    return df


def min_max_normalization(df):
    """
    最小値0、最大値１に正規化
        x' = x - min(x) / max(x) - min(x)
    :param df:
    :return:
    """
    df = pd.DataFrame(df)
    mm = preprocessing.MinMaxScaler()
    min_max = mm.fit_transform(df)
    return min_max


def zscore_normalize(x, m, std):
    """
    既に平均と標準偏差がわかっている場合の関数
    normalize : 面積0.5からzスコアを差し引いた値がnormalizeとして帰る。
                面積１とした時
    (例) : ~歳の平均体重が61.1kg, 標準偏差8.51, ~歳16万人いるとすれば、65kgは全国で何番目程か
            X=65, m=61.1, std=8.51
            160000*0.323=52000番目程度
    :param x: 観測値
    :param m: 平均
    :param std: 標準偏差
    :return:
    """
    h = x - m
    normalize = round(h / std, 2)
    probability = round(stats.norm.sf(x=normalize), 3)
    print(f'0 ~ {normalize}: 面積{0.5 - probability}')
    print(f'{normalize} ~ : 面積{probability}')

    mu = 0
    sigma = 1
    R = np.arange(-3, 3, 0.1)

    norm_pdf = stats.norm.pdf(x=R, loc=mu, scale=sigma)
    norm_pdf_max = np.max(norm_pdf)

    plt.figure(figsize=(11, 6))
    plt.plot(R, norm_pdf, lw=5, color="tab:cyan")
    plt.vlines(0, 0, norm_pdf_max + 0.02, color="black", lw=1, linestyle='--')
    plt.xticks(color="None")
    plt.yticks(color="None")
    plt.ylim(0, norm_pdf_max + 0.02)
    plt.text(-0.05, -0.03, 'μ', fontsize=12)
    plt.text(normalize, -0.03, f'{normalize}', fontsize=12, rotation=-0, horizontalalignment='center')
    plt.fill_between(R, norm_pdf, 0, where=(0 < R) & (R < normalize))
    plt.show()
    return probability


def zscore_normalize01(m, std, q=0.99, cc='cm', mm=0):
    """
    (例) 平均身長169.7cm、標準偏差5.6、柱の高さを決める時、この柱に頭をぶつける
         人の数を1%以内にしたい場合、柱の高さを何センチにする必要があるか？
    ppdによって0.5-0.01=0.49に近い値であるｚスコア2.33を取得する(正規分布表により)。
    平均から標準偏差の2.33倍いけばそれよりも右側の面積が0.01になるということなので、
    標準偏差は5.6である事がすでにわかっているのでzスコアと標準偏差を乗算し平均に加算する事で
    0~0.49のｚスコア領域をxcmに変換可能。
    :param m: 平均
    :param std: 標準偏差
    :param q: 指定する％、１％の場合0.99　5%の場合0.95
    :param cc: 使用単位
    :param mm: round丸目桁数
    :return:
    """
    z = stats.norm.ppf(q=q, loc=0, scale=1)
    x = std * z
    c = m + x
    pp = 1 - q
    return print(f'Zをxに変換 : {round(pp*100)}% : {round(c, mm)}{cc}')



if __name__ == "__main__":

    #lista = [107, 132, 120, 116, 130, 126, 116, 122]
    dff = pd.DataFrame({'target': [107, 132, 120, 116, 130, 126, 116, 122]})

    geometric_average(dff, 'target')

    Reconciliation(360, 180, 120)

    data = [15, 20, 45, 12, 8]
    deviation_sum_of_square(data)

    normalized_distribution_std(dff, target='target')

    min_max_normalization(dff)

    #zscore_normalize(65, 61.1, 8.51)

    zscore_normalize01(169, 5.6)

    unbased_varinace(dff)

    dff = pd.DataFrame({'targetA': [160, 170, 155, 165, 162, 158, 149, 179],
                        'targetB': [143, 145, 150, 155, 167, 160, 159, 159]})

    dispersion_add(dff, targetA='targetA', targetB='targetB')