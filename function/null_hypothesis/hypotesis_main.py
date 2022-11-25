import numpy as np
import pandas as pd
import scikit_posthocs as sp

from scipy import stats

pd.options.display.float_format = '{:,.4f}'.format
"""
パラメトリック検定は正規分布に従うデータに用いることできる検定
正規分布では平均と分散がパラメータであり、これらを用いて検定するためパラメトリック検定

平均と分散は比率あるいは間隔尺度のデータの代表値なので、
これらのデータでにしかパラメトリック検定は適用できません

パラメトリック検定を用いるか、ノンパラメトリック検定を用いるかを判断する
最初のステップは母集団の分布が正規分布であるかどうか


ノンパラメトリック検定は正規分布に従っていなくても使用できます。
この検定は母集団分布が不明であることを前提に検定

正規分布に従っていれば用いることができないかというとそうではなくて、
正規分布に従っていたとしても、母集団分布が不明であることにして用いることができます。
ただし、本来なら対立仮説を採択するはずが帰無仮説を採択する可能性
（これを第Ⅱ種の過誤という）が大きくなります。



"""


def check_normality_shapiro(data):
    """
    シャピロ-ウィルク検定：正規性の検定
    比率や間隔尺度のデータが正規分布に従っているか否かを知りたい時に用いる正規性検定の一つ
    つまり、データが正規分布にう従うかどうかの検定。
    ---------------------------------
    一般的に小さいサンプルに好まれる
    ---------------------------------
    H0 : 正規分布に従う　<-　正規分布でない
    H1 : 正規分布に従わない　<- 正規分布に従う
    :param data:
    :return:
    """
    t_normality, pval_normality = stats.shapiro(data)
    print(f"p_value : {pval_normality:,.4f}")
    if pval_normality < 0.05:
        print(f"H0:{pval_normality:,.4f} < 0.05 : 帰無仮説を棄却しました。データは正規分布ではありません。")
    else:
        print(f"H1:{pval_normality:,.4f} > 0.05 : 帰無仮説の棄却できません。データは正規分布です。")


def check_variance_homogeneity_levene(data1, *args):
    """
    分散の均一性をチェック
    Levene検定は、すべての入力サンプルが等しい分散を持つ母集団からのものであるという帰無仮説を検定
    母分散が等しいという帰無仮説を検定します（分散の均一性または等分散性と呼ばれます）。
    結果として得られるLeveneの検定のp値が有意水準（通常は0.05）よりも小さいと仮定する。
    Bartlettのテストをチェックすることも可能
    その場合、分散が等しい母集団からのランダムサンプリングに基づいて、
    得られたサンプル分散の差が発生する可能性は低くなります。
    ---------------------------------------------------------
    Leveneの検定には3つのバリエーションがあります。可能性とその推奨される使用法
    center=''
    ・'median'  :  歪んだ（非正規）分布に推奨>
    ・'mean'    :  対称で中程度の裾の分布に推奨されます。
    ・'trimmed' :  裾が重い分布に推奨されます。
    ---------------------------------------------------------
    H0: サンプルの分散は同じ
    H1: サンプルの分散は異なる　<つまり対になっていないのでパラメトリックバージョンテスト可能>
    :param data1:
    :param args:
    :return: 統計検定量、ｐ値　<float>
    """
    test_stat_var, pvalue_var = stats.levene(data1, *args)
    print(f"p_value: {pvalue_var:,.4f}")
    if pvalue_var < 0.05:
        print(f"H0:{pvalue_var:,.4f} < 0.05 : 帰無仮説を棄却しました。標本の分散は異なっています")
    else:
        print(f"H1:{pvalue_var:,.4f} > 0.05 : 帰無仮説を棄却できません。標本の分散は同じです。")


def ttest_ind(data1, data2):
    """
    ２つのスコア独立したサンプルの平均のT検定を計算する
    これは、２つの独立したサンプルが同一の平均(期待値)を持っているという帰無仮説の検定
    このテストは、母集団の分散がデフォルトで同じであることを前提とする。
    --------------------------------------
    正規検定　 : H1
    分散均一性 : H1
    である場合に使用可能。
    --------------------------------------
    equal_var= bool : def(True) 母分散が等しいと仮定する標準の独立した２サンプル実行
                    　False     の場合、ウェルチのT検定を実行。
                                母分散が等しいとは想定していない
    nan_policy      : 入力にnanが含まれている場合の処理方法を定義する。
                      def 'propagate' : nanを返す
                          'raise'     : エラーをスローする
                          'omit'      : nan値を無視して計算を実行する。

    :param data1:
    :param data2:
    :return: 統計、ｐ値　計算されたt統計、p値
    """
    ttest, p_value = stats.ttest_ind(data1, data2)
    print(f"p_value{p_value:,.6f}")
    print(f"仮説は片側検定になるので　p_value/2を使用して片側のみを使用する。{p_value / 2:,.6f}")
    if p_value / 2 < 0.05:
        print(f"H0: {p_value / 2:,.6f} < 0.05 : よって帰無仮説を棄却する。")
        print("帰無仮説を棄却できるので、仮説は「有意性があり」正しいといえる。")
        if p_value / 2 < 0.01:
            print(f"H0: {p_value / 2:,.6f} < 0.01 : 「高度に有意性がある」ともいえる")
            print("よって仮説はより正しいと言える。")
        else:
            print(f"H0: {p_value / 2:,.6f} > 0.01 ：であるので「高度に有意性があるとは」言えない")
    else:
        print(f"H1: {p_value / 2:,.6f)}帰無仮説の棄却に失敗しました。仮説が正しいとは言えません")


def ttest_rel(data1, data2):
    """
    スコアの2つの関連サンプルaとbのt検定を計算
    ２つの関連している(対)サンプルまたは繰り返されるサンプルの平均（期待）値が同じであるという帰無仮説の検定
    -------------------------------------------
    (例) データは同じ個人から収集され、仮定が満たされているため、
    データはペアになり、従属 t検定 を使用
    :param data1:
    :param data2:
    :return:
    """
    ttest, p_value = stats.ttest_rel(data1, data2)
    print(f"p_value{p_value:,.6f}")
    print(f"仮説は片側検定になるので　p_value/2を使用して片側のみを使用する。{p_value / 2:,.6f}")
    if p_value / 2 < 0.05:
        print(f"H0: {p_value / 2:,.6f} < 0.05 : よって帰無仮説を棄却する。")
        print("帰無仮説を棄却できるので、仮説は「有意性があり」正しいといえる。")
        if p_value / 2 < 0.01:
            print(f"H0: {p_value / 2:,.6f} < 0.01 : 「高度に有意性がある」ともいえる")
            print("よって仮説はより正しいと言える。")
        else:
            print(f"H0: {p_value / 2:,.6f} > 0.01 ：であるので「高度に有意性があるとは」言えない")
    else:
        print(f"H1: {p_value / 2:,.6f}帰無仮説の棄却に失敗しました。仮説が正しいとは言えません")


def f_oneway_anova(*args):
    """
    ・<パラメトリックANOVA>
    一元配置分散分析を実行します。
    一元配置分散分析は、2つ以上のグループが同じ母平均を持つという帰無仮説を検定します。
    テストは、サイズが異なる可能性のある2つ以上のグループのサンプルに適用されます
    ---------------------------------------------
    ANOVA検定には、関連するp値が有効であるために満たされなければならない重要な仮定
    ・サンプルは独立しています。
    ・各サンプルは、正規分布の母集団からのものです。
    ・グループの母標準偏差はすべて等しい。この特性は等分散性として知られています。
    ---------------------------------------------
    もし、これらの仮定が与えられたデータセットに対して正しくない場合、
    多少検出力が落ちるものの、Kruskal-Wallis H-検定 (scipy.stats.kruskal) や
    Alexander-Govern 検定 (scipy.stats.alexandergovern) が使用できる可能性があります。

    :param args:
    :return: F統計、F分布からの関連するｐ値
    """
    F, p_value = stats.f_oneway(*args)
    print(f"p_value : {p_value:,.6f}")
    if p_value < 0.05:
        print(f"H0 : {p_value:,.6f} < 0.05 : 帰無仮説を棄却する")
        print("帰無仮説を棄却できるので、仮説は「有意性があり」正しいといえる。")
        if p_value < 0.01:
            print(f"H0: {p_value:,.6f} < 0.01 : 「高度に有意性がある」ともいえる")
            print("よって仮説はより正しいと言える。")
    else:
        print(f"H1 : {p_value:,.6f} > 0.05 : 帰無仮説を棄却できません")


def sc_posthoc_ttest(*args):
    """
    <パラメトリックANOVA-ペアワイズ>

    独立したグループの多重比較のためのペアワイズT検定。
    ・ペアワイズ比較を行うために、パラメトリックANOVAの後に使用できます。
    ------------------------------------------------------
    p_adjust : bonferroni       ワンステップ補正
               sidak            ワンステップ補正
               holm-sidak       Sidak補正を使用したステップダウン方法
               holm             ボンフェローニ補正を使用したステップダウン方法
               simes-hochberg   ステップアップ法(独立)
               hommel           Simesテストに基づくクローズド法(非負)
               fdr_bh           Benjamini/Hochberg(非負)
               fdr_by           Benjamini/Yekutieli(負)
               fdr_tsbh         2段階fdr補正(非負)
               fde_tsby         2段階fdr補正(非負)
    ------------------------------------------------------
    :param args: array or dataframe (現状可搬性を重視してarrayのみ使用可能)
    :return: p値　round()
    """
    posthoc_df = sp.posthoc_ttest([*args], equal_var=True, p_adjust="bonferroni")
    group_name = [f"Target_{x}" for x in range(len(posthoc_df.columns))]
    posthoc_df.columns = group_name
    posthoc_df.index = group_name
    posthoc_df.style.applymap(lambda x: "background-color:violet" if x < 0.05 else "background-color: black")
    return print(round(posthoc_df, 6))


def sc_posthoc_mannwhitney(*args):
    """
    <ノンパラメトリックANOVA-ペアワイズ比較>

    マンホイットニー順位検定とのペアワイズ比較
    ノンパラメトリック版ポストホックテスト
    ---------------------------------------
    (例)
    データは正規分布ではない
    平均顧客獲得数のうち少なくとも1つが異なっている
    :param args:
    :return:
    """
    posthoc_df = sp.posthoc_ttest([*args], equal_var=True, p_adjust="bonferroni")
    group_name = [f"Target_{x}" for x in range(len(posthoc_df.columns))]
    posthoc_df.columns = group_name
    posthoc_df.index = group_name
    posthoc_df.style.applymap(lambda x: "background-color:violet" if x < 0.05 else "background-color: black")
    return print(posthoc_df)


def sc_posthoc_wilcoxon(*args):
    posthoc_df = sp.posthoc_wilcoxon([*args], p_adjust="holm")
    group_name = [f"Target_{x}" for x in range(len(posthoc_df.columns))]
    posthoc_df.columns = group_name
    posthoc_df.index = group_name
    posthoc_df.style.applymap(lambda x: "background-color:violet" if x < 0.05 else "background-color: black")
    return print(posthoc_df)


def mann_whitney_u(data1, data2, alternative="two-sided"):
    """
    <ノンパラメトリック>
    2つの独立したサンプルでマンホイットニーのUランク検定を実行します。
    マンホイットニーU検定は、サンプルxの基礎となる分布がサンプルyの基礎となる
    分布と同じであるという帰無仮説のノンパラメトリック検定です
    ------------------------------------------------------
    use_continuity bool   : 連続性補正（1/2）を適用するかどうか。
                            メソッドが'asymptotic';の場合、デフォルトはTrue
    alternative           : 対立仮説を定義、デフォルトは'two-sided'
                            F（u）とG（u）を、それぞれxとyの基礎となる分布の累積分布関数
                            次に、次の対立仮説が利用可能
                            'two-sided'
                            分布は等しくありません。つまり、
                            少なくとも1つのuに対してF（u）≠G（u）
                            'less'
                            xの基礎となる分布は、 yの基礎となる分布よりも確率的に小さくなります。
                            つまり、すべてのuについてF（u）> G（u）
                            'greater'
                            xの基礎となる分布は、 yの基礎となる分布よりも確率的に大きくなります。
                            つまり、すべてのuについてF（u）<G（u）

    :param data1: x data
    :param data2: y data
    :param alternative: 対立仮説定義
    :return:
    """
    u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative=alternative)
    print(f"p_value : {p_value:,.6f}")
    if p_value < 0.05:
        print(f"H0 : {p_value:,.6f} < 0.05 : 帰無仮説を棄却する")
        print("帰無仮説を棄却できるので、仮説は「有意性があり」正しいといえる。")
        if p_value < 0.01:
            print(f"H0: {p_value:,.6f} < 0.01 : 「高度に有意性がある」ともいえる")
            print("よって仮説はより正しいと言える。")
    else:
        print(f"H1 : {p_value:,.6f} > 0.05 : 帰無仮説を棄却できません")


def levene_test(*args):
    """
    等分散性についてLevene検定を実行します。
    Levene検定は、すべての入力サンプルが等しい分散を持つ母集団からのものであるという
    帰無仮説を検定します 。
    Levene bartlettの検定は、正規性からの有意な偏差がある場合のbartlettの検定の代替です。
    -----------------------------------------------
    center :テストで使用するデータの機能, デフォルトmedian
            mean
            対称で中程度の裾の分布に推奨
            median
            歪んだ（非正規）分布に推奨>
            trimmed
            裾が重い分布に推奨
    -----------------------------------------------
    平均を使用したテストバージョンは、Leveneの元の記事で提案されましたが、
    中央値とトリム平均は、ブラウンフォーサイス検定によって研究されており、
    ブラウンフォーサイス検定
    ----------------------------------------------
    p値が小さいということは、母集団の分散が等しくないことを示す

    :param args: data... n
    :return: stat, p_value
    """
    stat, p_value = stats.levene(*args)
    print(f"p_value : {p_value:,.6f}")
    if p_value < 0.05:
        print(f"H0:{p_value:,.4f} < 0.05 : 帰無仮説を棄却しました。標本の分散は異なっています")
    else:
        print(f"H1:{p_value:,.4f} > 0.05 : 帰無仮説を棄却できません。標本の分散は同じです。")


def nonpara_kruskal_wallis_h(*args):
    """
    <ノンパラメトリックANOVA>

    独立したサンプルのクラスカル・ウォリスH検定を計算します。
    Kruskal -Wallis H検定は、すべてのグループの母集団の中央値が等しいという帰無仮説を検定
    -------------------------------------------------
    テストは、サイズが異なる可能性のある2つ以上の独立したサンプルで機能します。
    帰無仮説を棄却しても、どのグループが異なるかを示すものではないことに注意してください。
    どのグループが異なるかを判断するには、グループ間の事後比較が必要
    :param args:
    :return: H_stat, p_value
    """
    H, p_value = stats.kruskal(*args)
    if p_value < 0.05:
        print(f"H0 : {p_value:,.6f} < 0.05 : 帰無仮説を棄却する")
        print("帰無仮説を棄却できるので、仮説は「有意性があり」正しいといえる。")
        if p_value < 0.01:
            print(f"H0: {p_value:,.6f} < 0.01 : 「高度に有意性がある」ともいえる")
            print("よって仮説はより正しいと言える。")
    else:
        print(f"H1 : {p_value:,.6f} > 0.05 : 帰無仮説を棄却できません")


def nonpara_wilcoxon(data1, data2, alternative='less'):
    """
    <ノンパラメトリック検定>
    ペア検定
    符号付き順位検定でウィルコクソンを計算
    符号付き順位検定のWilcoxは、2つの関連するペアの標本が同じ分布からのものであるという帰無仮説を検定

    差x--yの分布がゼロに関して対称であるかどうかをテスト
    ---------------------------------------
    zero_method : wilcox (defo)
                  デフォルトのゼロ差をすべて破棄します。
                  pratt
                  ランク付けプロセスにゼロの差を含めますが、ゼロのランクを下げます。
                  zsplit
                  ランク付けプロセスにゼロの差を含め、ゼロのランクを正と負のランクに分割
    ---------------------------------------
    (例)
    正規性の仮定は満たされていません
    ペア検定のノンパラメトリックバージョン、
    つまりウィルコクソン符号順位検定を使用する必要
    :param data1:
    :param data2:
    :param alternative:
    :return:
    """
    ttest, p_value = stats.wilcoxon(data1, data2, alternative=alternative)
    if alternative == 'two-sided':
        if p_value < 0.05:
            print(f"H0 : {p_value:,.6f} < 0.05 : 帰無仮説を棄却する")
            print("帰無仮説を棄却できるので、仮説は「有意性があり」正しいといえる。")
            if p_value < 0.01:
                print(f"H0: {p_value:,.6f} < 0.01 : 「高度に有意性がある」ともいえる")
                print("よって仮説はより正しいと言える。")
        else:
            print(f"H1 : {p_value:,.6f} > 0.05 : 帰無仮説を棄却できません")
    elif alternative == 'less':
        if p_value < 0.05:
            print(f"H0 : {p_value:,.6f} < 0.05 : 帰無仮説を棄却する")
            print("帰無仮説を棄却できるので、仮説は「有意性があり」正しいといえる。")
            if p_value < 0.01:
                print(f"H0: {p_value:,.6f} < 0.01 : 「高度に有意性がある」ともいえる")
                print("よって仮説はより正しいと言える。")
        else:
            print(f"H1 : {p_value:,.6f} > 0.05 : 帰無仮説を棄却できません")
    else:
        print('alternative ERROR : two-sided, less, greater')


def nonpara_friedman(*args):
    """
    <ノンパラメトリックANOVA>
    ペア検定
    ペアデータにはノンパラメトリックバージョンのANOVAを使用する必要
    --------------------------------------------------------
    (例１)
    さまざまな方法で取得された測定値間の一貫性をテストするためによく使用
    たとえば、2つの測定手法が同じ個人のセットで使用される場合、フリードマン検定を使用して、
    2つの測定手法が一貫しているかどうかを判断
    --------------------------------------------------------
    (例２)
    3つのグループがありますが、正規性の仮定に違反(どれか１でも)
    精度スコアは同じテストセットから取得
    ペアデータにはノンパラメトリックバージョンのANOVAを使用する必要
    :param args:
    :return:
    """
    stat, p_value = stats.friedmanchisquare(*args)
    if p_value < 0.05:
        print(f"H0 : {p_value:,.6f} < 0.05 : 帰無仮説を棄却する")
        print("帰無仮説を棄却できるので、仮説は「有意性があり」正しいといえる。")
        if p_value < 0.01:
            print(f"H0: {p_value:,.6f} < 0.01 : 「高度に有意性がある」ともいえる")
            print("よって仮説はより正しいと言える。")
    else:
        print(f"H1 : {p_value:,.6f} > 0.05 : 帰無仮説を棄却できません")


def chi2_contingency(data, alpha=0.01,  correction=False):
    """
    カイ二乗連続確率変数
    :param data:
    :param alpha:
    :param correction:
    :return:
    """
    chi2, p, dof, ex = stats.chi2_contingency(data, correction=correction)
    print(f"予想される頻度 : {np.round(ex, 2)}")
    print(f"検定統計量 : {chi2:,.4f}")
    print(f"自由度 : {dof}")
    print(f"P_Value : {p:,.4f}")
    alpha = alpha
    df = (data.shape[1] - 1) * (data.shape[0] - 1)
    c_test = stats.chi2.ppf((1-alpha), df)
    print(f"臨界統計量: {c_test:,.4f}")
    if p > alpha and chi2 < c_test:
        print(f"P_Value : {p:,.4f} > α：{alpha} | 検定統計量 : {chi2:,.4} < 臨界統計量 : {c_test:,.4}")
        print("H0 : 帰無仮説を棄却できません。")
    elif p > alpha:
        print(f"P_Value : {p:,.4f} > α：{alpha} : 帰無仮説を棄却できません")
        print(f"検定統計量 : {chi2:,.4} = 臨界統計量 : {c_test:,.4}")
    elif p > alpha and chi2 > c_test:
        print(f"P_Value : {p:,.4f} < α：{alpha} | 検定統計量 : {chi2:,.4} > 臨界統計量 : {c_test:,.4}")
        print(f"H0 : 帰無仮説を棄却しました。")
    elif p > alpha:
        print(f"P_Value : {p:,.4f} < α：{alpha} : 帰無仮説を棄却しました。")
        print(f"検定統計量 : {chi2:,.4} = 臨界統計量 : {c_test:,.4}")
    else:
        print("Error")


if __name__ == '__main__':

    sync = np.array([94., 84.9, 82.6, 69.5, 80.1, 79.6, 81.4, 77.8, 81.7, 78.8, 73.2,
                     87.9, 87.9, 93.5, 82.3, 79.3, 78.3, 71.6, 88.6, 74.6, 74.1, 80.6])
    asyncr = np.array([77.1, 71.7, 91., 72.2, 74.8, 85.1, 67.6, 69.9, 75.3, 71.7, 65.7, 72.6, 71.5, 78.2])

    check_normality_shapiro(sync)
    check_normality_shapiro(asyncr)
    check_variance_homogeneity_levene(sync, asyncr)
    ttest_ind(sync, asyncr)

    only_breast = np.array([794.1, 716.9, 993., 724.7, 760.9, 908.2, 659.3, 690.8, 768.7,
                            717.3, 630.7, 729.5, 714.1, 810.3, 583.5, 679.9, 865.1])
    only_formula = np.array([898.8, 881.2, 940.2, 966.2, 957.5, 1061.7, 1046.2, 980.4,
                             895.6, 919.7, 1074.1, 952.5, 796.3, 859.6, 871.1, 1047.5,
                             919.1, 1160.5, 996.9])
    both = np.array([976.4, 656.4, 861.2, 706.8, 718.5, 717.1, 759.8, 894.6, 867.6,
                     805.6, 765.4, 800.3, 789.9, 875.3, 740., 799.4, 790.3, 795.2,
                     823.6, 818.7, 926.8, 791.7, 948.3])
    check_normality_shapiro(only_breast)
    check_normality_shapiro(only_formula)
    check_normality_shapiro(both)
    check_variance_homogeneity_levene(only_breast, only_formula, both)
    f_oneway_anova(only_breast, only_formula, both)
    sc_posthoc_ttest(only_breast, only_formula, both)

    test_team = np.array([6.2, 7.1, 1.5, 2, 3, 2, 1.5, 6.1, 2.4, 2.3, 12.4, 1.8, 5.3, 3.1, 9.4, 2.3, 4.1])
    developer_team = np.array([2.3, 2.1, 1.4, 2.0, 8.7, 2.2, 3.1, 4.2, 3.6, 2.5, 3.1, 6.2, 12.1, 3.9, 2.2, 1.2, 3.4])
    check_normality_shapiro(test_team)
    check_normality_shapiro(developer_team)
    check_variance_homogeneity_levene(test_team, developer_team)
    mann_whitney_u(test_team, developer_team)

    youtube = np.array([1913, 1879, 1939, 2146, 2040, 2127, 2122, 2156, 2036, 1974, 1956,
                        2146, 2151, 1943, 2125])
    instagram = np.array([2305., 2355., 2203., 2231., 2185., 2420., 2386., 2410., 2340.,
                          2349., 2241., 2396., 2244., 2267., 2281.])
    facebook = np.array([2133., 2522., 2124., 2551., 2293., 2367., 2460., 2311., 2178.,
                         2113., 2048., 2443., 2265., 2095., 2528.])
    check_normality_shapiro(youtube)
    check_normality_shapiro(instagram)
    check_normality_shapiro(facebook)
    check_variance_homogeneity_levene(youtube, instagram, facebook)
    levene_test(youtube, instagram, facebook)
    nonpara_kruskal_wallis_h(youtube, instagram, facebook)
    sc_posthoc_mannwhitney(youtube, instagram, facebook)

    test_results_before_diet = np.array(
        [224, 235, 223, 253, 253, 224, 244, 225, 259, 220, 242, 240, 239, 229, 276, 254, 237, 227])
    test_results_after_diet = np.array(
        [198, 195, 213, 190, 246, 206, 225, 199, 214, 210, 188, 205, 200, 220, 190, 199, 191, 218])
    check_normality_shapiro(test_results_before_diet)
    check_normality_shapiro(test_results_after_diet)
    ttest_rel(test_results_before_diet, test_results_after_diet)

    piedpiper = np.array(
        [4.57, 4.55, 5.47, 4.67, 5.41, 5.55, 5.53, 5.63, 3.86, 3.97, 5.44, 3.93, 5.31, 5.17, 4.39, 4.28, 5.25])
    endframe = np.array(
        [4.27, 3.93, 4.01, 4.07, 3.87, 4., 4., 3.72, 4.16, 4.1, 3.9, 3.97, 4.08, 3.96, 3.96, 3.77, 4.09])
    check_normality_shapiro(piedpiper)
    check_normality_shapiro(endframe)
    nonpara_wilcoxon(endframe, piedpiper, alternative='less')

    method_A = np.array([89.8, 89.9, 88.6, 88.7, 89.6, 89.7, 89.2, 89.3])
    method_B = np.array([90.0, 90.1, 88.8, 88.9, 89.9, 90.0, 89.0, 89.2])
    method_C = np.array([91.5, 90.7, 90.3, 90.4, 90.2, 90.3, 90.2, 90.3])
    check_normality_shapiro(method_A)
    check_normality_shapiro(method_B)
    check_normality_shapiro(method_C)
    levene_test(method_A, method_B, method_C)
    nonpara_friedman(method_A, method_B, method_C)
    sc_posthoc_wilcoxon(method_A, method_B, method_C)

    obs = np.array([[53, 23, 30, 36, 88], [71, 48, 51, 57, 203]])
    chi2_contingency(obs)

