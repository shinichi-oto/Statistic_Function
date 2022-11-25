import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats


def input_operation(df, random_state=42):
    """
    テスト(未完成)
    データフレームのランダムサンプリング
    :param random_state:
    :param df:
    :return: DataFrame
    """
    while True:
        ratio = input("Use sample-size ratio or n is int >>")
        if ratio == "ratio":
            print(f"データサイズ : index{len(df)}")
            frac = float(input("サンプル比率を指定してください。： float >>"))
            if frac:
                df = df.sample(frac=frac, random_state=random_state)
            else:  # test
                print("比率の指定が不正です、floatで比率を指定してください。")
        elif ratio == "n":
            print(f"データサイズ : index{len(df)}")
            n = int(input("サンプルサイズを指定してください。： int >>"))
            if n:  # test
                df = df.sample(n=n, random_state=random_state)
        else:
            print("入力が不正です。入力はbool値のみ受け付け可能。True or False")
        return df


def interval_estimation_parameter(df, sigma2=5):
    """
    <母平均の区間推定>
    母分散がわかっている場合の推定
    -------------------------------------------------
    人間の手によって作成された現実世界のデータに使用される。
    現実世界の大部分のデータは母分散がわからない。
    -------------------------------------------------
    (例) : 機械などのデータで、母集団がわかる場合に使用
    生徒が500人、母集団500人の点数分布、などにも使用。
    :param df:　DataFrame 指定したColumnのみ対応、事前に抜き出す必要あり
    :param sigma2:　母分散
    :return:　母平均区間推定
    """
    mu = df.mean()[0]
    n = len(df)
    up = mu + round(1.96 * np.sqrt(sigma2 ** 2 / n), 1) # noqa
    dw = mu - round(1.96 * np.sqrt(sigma2 ** 2 / n), 1)
    return print(f"[{dw}, {up}]")


def interval_estimation_parameter_stats(df, sigma=5):
    """
    母平均区間推定
    *母分散がわかっている場合に使用可能な関数。
    stats関数使用Version
    --------------------------------------
    :param df: DataFrame 指定したColumnのみ対応、事前に抜き出す必要あり
    :param sigma: 母分散
    :return: 母平均区間推定
    """
    mu = df.mean()[0]
    n = len(df)
    bottom, up = stats.norm.interval(0.95, loc=mu, scale=np.sqrt(sigma**2 / n)) # noqa
    return print(f"[{round(bottom, 1)}, {round(up, 1)}]")


# noinspection PyGlobalUndefined
def interval_estimation_variance_unknown(df):
    """
    母平均区間推定
    ＊母分散が未知の場合に使用できる関数
    stats関数未使用
    :param df: 対象とするDataFrame　Column：１にしなければ動作しない。
    :return: 母平均区間推定
    """
    s = df.sum()[0]
    df['rap'] = df**2
    rap = df['rap'].sum()
    n = len(df)
    sample_mean = s / n
    variance = (n * rap - s**2) / (n * (n - 1))

    global bottom, up
    if 30 >= n:
        bottom = sample_mean - 1.96 * np.sqrt(variance / n)
        up = sample_mean + 1.96 + np.sqrt(variance / n)
    elif 30 < n:
        bottom = sample_mean - 2.262 * np.sqrt(variance / n)
        up = sample_mean + 2.262 * np.sqrt(variance / n)
    return print(f"[{bottom:,.2f}, {up:,.2f}]")


def interval_estimation_variance_unknown_stats(df):
    """
    母平均区間推定
    * 母分散が未知の場合に使用可能な関数。
    stats関数使用。
    :param df: 対象とするColumn : 1
    :return: 母平均区間推定
    """
    s = df.sum()[0]
    df['rap'] = df**2
    rap = df['rap'].sum()
    n = len(df)
    sample_mean = s / n
    deg_of_free = n - 1
    variance = (n * rap - s**2) / (n * deg_of_free)
    if n < 30:
        # t分布
        bottom, up = stats.t.interval(0.95, deg_of_free, loc=sample_mean, scale=np.sqrt(variance / n))
        print(f"データ数:{n} < 30 の為、t分布使用 : 母平均区間推定 [{bottom:,.2f}, {up:,.2f}]")
    elif n >= 30:
        # 正規分布
        bottom2, up2 = stats.norm.interval(alpha=0.95, loc=sample_mean, scale=np.sqrt(variance / n))
        print(f"データ数:{n} >= 30 の為、正規分布使用 : 母平均区間推定 [{bottom2:,.2f}, {up2:,.2f}]")


def interval_estimation_ratio(numerator=40, denominator=160, k=1.96):
    """
    母比率の区間推定
    母比率の区間推定を求める場合、標本の数が十分大きいときは近似的に標準正規分布に従う
    -------------------------------------------------
    この関数は元のデータを使用せずに推定可能。
    -------------------------------------------------
    (例)
    ある地域の有権者160人を調査したところ、40人がA党を支持すると回答。
    この地域のA党の支持率を信頼係数95%で推定する。
    160 = denominator, 40 = numerator,
    k=95% = 1.96(標本数が多い場合近似的に正規分布なので標準正規分布の係数を使用可能。)
    :param numerator: 分子
    :param denominator: 分母
    :param k: 信頼係数　95% = 1.96 : 90% = 1.65
    :return: 比率信頼区間
    """
    sample_ratio = numerator / denominator
    bottom = sample_ratio - k * np.sqrt((sample_ratio * (1 - sample_ratio)) / denominator)
    up = sample_ratio + k * np.sqrt((sample_ratio * (1 - sample_ratio)) / denominator)
    print(f'[{bottom:,.4}, {up:,.4f}]')
    print(f'[{bottom*100:,.1f}%, {up*100:,.1f}%]')


def interval_estimation_ratio_stats(alpha=0.95, numerator=40, denominator=160):
    """
    母比率の区間推定
    --------------------------------
    デフォルト値で
    (例１)ある地域の街頭のアンケートで、通行人160人にパンを自宅で作ったことがあるかと尋ね
    40人が作ったことがあると回答した。
    信頼係数95%でパンを作ったと回答した人の割合の信頼区間を求める
    ・ある地域でパンを作ったことがある確率は、95%の確率で、およそ18%~31%だろうという事が推定できる。
    --------------------------------
    (例２)調査対象として、300世帯の標本のうち、120世帯がある商品を使用していたとする
    この場合の標本比率は0.40。これ計算すると、信頼係数95%では、35.2~44.6%の間にあると言える。
    これは10%以上の開きがあるが、つまり95%の確率で、およそ35~44%という事である。
    --------------------------------
    :param alpha:　信頼係数
    :param numerator: 分子
    :param denominator: 分母
    :return: 母比率の区間推定値
    """
    sample_ratio = numerator / denominator
    bottom, up = stats.binom.interval(alpha=alpha, n=denominator, p=sample_ratio, loc=0)
    print(f'{alpha*100}%の確率で[{(bottom / denominator)*100:,.2f}% ~ {(up / denominator)*100:,.2f}%]')


def interval_estimation_ratio_difference(alpha=0.95, numerator_1=80, denominator_1=200,
                                         numerator_2=60, denominator_2=300):
    z = round(((1 - alpha) / 2), 3)
    # add alphaからz値を取得する、stats関数以下を使用する。
    sample_ratio_1 = numerator_1 / denominator_1
    sample_ratio_2 = numerator_2 / denominator_2
    dw = (sample_ratio_1 - sample_ratio_2) - 1.96 * np.sqrt(((sample_ratio_1 * (1 - sample_ratio_1)) / denominator_1) +
                                                         ((sample_ratio_2 * (1 - sample_ratio_2)) / denominator_2))
    up = (sample_ratio_1 - sample_ratio_2) + 1.96 * np.sqrt(((sample_ratio_1 * (1 - sample_ratio_1)) / denominator_1) +
                                                         ((sample_ratio_2 * (1 - sample_ratio_2)) / denominator_2))
    print(f"[{dw}, {up}]")


def samples_estimation_mean(error_ratio=0.9, k=2.58, std=3):
    """
    <標本の数をどの程度にすればよいか>
    ・母平均推定の標本数
    ・何件のデータを取ればよいのかという問題に対応する。
    基本的に標本数はstd、つまり標準偏差の値によって決定される。
    --------------------------------------
    (例１)
    紙の長さの誤差の標準偏差は、今までの経験から約3メートル
    この紙の長さの99%信頼区間を求める。その時、誤差の比率を0.9にする。
    この関数のデフォルトを使用すれば、約74個の紙についての標本を抜き
    出せばよい事になる。
    --------------------------------------
    k : 90% = 1.65
        95% = 1.96
        99% = 2.58
    :param error_ratio:
    :param k:
    :param std:
    :return:
    """
    nos = ((k / error_ratio) * std)**2
    print(nos)


def samples_estimation_ratio(sample_ratio=0.34, error_ratio=0.02, k=1.96):
    """
    <母比率推定の標本数>
    母比率の区間推定における標本の大きさを決定する。
    誤差の比率に注目し、標本の数を決定する。
    ---------------------------------------------
    (例１)
    内閣支持率34％。支持率を95%信頼区間で推定したとき、誤差を2%以内におさえる。
    標本数はいくつにすればよいか？
    予想母比率は=0.34, 誤差比率=0.02, k=1.96(95%)
    ---------------------------------------------
    (例２)
    種子の発芽する確率を、標本抽出により区間推定する。
    信頼係数90%で誤差を5%以内におさえるには、何粒実験しなければならないか？
    誤差比率0.05, k=1.65(90%)
    この場合、標本比率が不明なので、計算には標本比率を利用しない。ｎを選択する。
    :param sample_ratio: 標本比率
    :param error_ratio: 誤差比率
    :param k: 信頼係数 , 95% = 1.96, 90% = 1.65
    :return: Number of samples 標本数
    """
    while True:
        sr = input('標本比率が予測できる状態ですか？: [y, n]')
        if sr == 'y':
            nos = sample_ratio * (1 - sample_ratio) * (k / error_ratio)**2
            print(f"{nos:,.6}")
            break
        elif sr == 'n':
            nos = (1/4) * (k / error_ratio)**2
            print(f'{nos:,.6f}')
            break
        else:
            print("入力値が不正です。やり直してください。[y, n]")


def sample_size_pp(alpha=0.95, pp=0.10, breadth=0.05):
    """
    <妥当な標本サイズの取得>
    <誤差をn%に抑える為のサンプルサイズの取得>
    ---------------------------------------------------------
    (例１)信頼係数95%。番組の視聴率の信頼区間の誤差(幅)を5%以下にするには、
    何人以上の人にアンケートが必要か？ただし事前調査で視聴率は10%以下である事がわかっている
    ・視聴率を推定値として母比率に代入する。幅は5%以下であればよいのでbreadth=0.05
    :param alpha: 信頼係数 : 係数を指定するのみでｋの値が取得可能
    :param pp: population proportion　母比率　(例)事前に視聴率が10%以下である=0.10
    :param breadth: 幅　: 誤差をｎ%以内に抑える為の指定
    :return: 妥当なサンプルサイズ
    """
    bottom, up = stats.norm.interval(alpha, loc=0, scale=1)
    sample_size = (2 * up * np.sqrt(pp * (1 - pp)) * (1 / breadth))**2
    print(f'妥当なSample_Size : {sample_size:,.2f}')


def sample_size_not_pp(alpha=0.95, p_hat=0.5, breadth=0.04):
    """
    <母比率ppがわからない場合のサンプルサイズ取得>
    -------------------------------------------
    p_hatを推定値として使用する。ppが問題から取得できない場合、p_hat=0.5を使用する
    (理由)信頼区間が最大になるのはp_hat=0.5の時で、実際の推定値ｐがどのような値であっても
    それ以上信頼区間の幅が大きくなる事はない。
    母比率の信頼区間は、点推定の比率p_hat=0.5の時に最も広くなる。
    信頼区間の幅は、p_hatについて二次関数の部分があり、この部分は、サンプルサイズと
    信頼係数が固定された場合、p_hat=1/2の時に最大値1/4を取ることになる。
    -------------------------------------------
    p_hat = 0.1, 0.3, 0.5, 0.7, 0.99
    :param alpha:
    :param p_hat:
    :param breadth:
    :return:
    """
    bottom, up = stats.norm.interval(alpha, loc=0, scale=1)
    sample_size = (2 * up * np.sqrt(p_hat * (1 - p_hat)) * (1 / breadth))**2
    print(f'必要なサンプルサイズ：　{sample_size:,.0f}')


if __name__ == '__main__':

    df2 = pd.DataFrame([60, 70, 80, 80, 80, 90, 100, 80, 70, 90])

    interval_estimation_parameter(df2)
    interval_estimation_parameter_stats(df2, 5)

    df3 = pd.DataFrame([3210, 2800, 2910, 3340, 3520, 3190, 2680, 2900, 3250, 2900])
    interval_estimation_variance_unknown(df3)
    df4 = pd.DataFrame([3210, 2800, 2910, 3340, 3520, 3190, 2680, 2900, 3250, 2900])
    interval_estimation_variance_unknown_stats(df4)

    interval_estimation_ratio(40, 160)
    interval_estimation_ratio_stats()

    samples_estimation_mean()
    #samples_estimation_ratio()

    #samples_estimation_ratio(error_ratio=0.05, k=1.65)

    sample_size_pp()
    sample_size_not_pp()

    interval_estimation_ratio_difference()

    # z = 0.025 = 1.96 この0.025はｚ表
    a = (0.4 - 0.2) - 1.96 * np.sqrt(0.4 * (1 - 0.4)/200 + 0.2 * (1 - 0.2) / 300)
    print(a)