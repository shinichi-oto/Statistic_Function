import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from scipy import stats
from scipy.stats import chi2
from statsmodels.stats import weightstats as statsmd
from statsmodels.formula.api import ols


def one_sample_t_test(data):
    """
    データセットが正規分布に従い、未知の分散を持つ可能性がある場合に使用
    T検定は、母集団に適用可能な仮定の検定を可能にする仮説検定ツールとして使用されます。

    1標本t検定：1標本t検定は、標本平均が既知または仮説の母平均と統計的に異なるかどうかを判別します。
    One Sample t Testは、パラメトリック検定です。
    ----------------------------------------
    (例)：平均年齢~歳かどうかを確認する
    H_{0} : 平均年齢は30である
    H_{1} : 平均年齢は30ではない
    :param ages:
    :return:
    """
    ages_mean = np.mean(data)
    print(f'ages_mean : {ages_mean}')

    t_test, t_pval = stats.ttest_1samp(data, 30)
    print(t_test)

    if t_pval < 0.05:
        print(f"帰無仮説$H_{0}$を棄却 :  {round(t_pval, 4)} < 0.05")
    else:
        print(f"帰無仮説を棄却できません : {round(t_pval, 4)} > 0.05")


def two_sampled_t_test(data, target_1='', target_2=''):
    """
    インデペンデントサンプルt検定または、2標本t検定は、
    関連する母平均が有意に異なるという統計的証拠があるかどうかを
    判断するために、2つの独立したグループの平均を比較します。
    独立サンプルt検定は、パラメトリック検定です。このテストは、
    次のようにも知られています。独立t検定。
    ---------------------------------------------------------
    (例)　week1とweek2の間に関連性はあるか
    H_{0} : 関連はない
    H_{1} : 関連がある
    :param data:
    :param target_1:
    :param target_2:
    :return:
    """
    #data = pd.DataFrame(data)
    #data1 = np.array(data[target_1])
    #data2 = np.array(data[target_2])
    # test
    data1 = data
    data2 = data
    t_test, t_pval = stats.ttest_ind(data1, data2)

    if t_pval < 0.05:
        print(f"帰無仮説$H_{0}$を棄却 :  {round(t_pval, 4)} < 0.05")
    else:
        print(f"帰無仮説を棄却できません : {round(t_pval, 4)} > 0.05")


def paired_t_test(data):
    """
    対応のあるサンプルのt検定：-対応のあるサンプルのt検定は、従属サンプルのt検定とも呼ばれます。
    これは、2つの関連する変数間の有意差を検定する単変量検定です。
    この例は、治療、状態、または時点の前後に個人の血圧を収集する場所の場合です。

    H0：-2つのサンプル間の差が0であることを意味します
    H1：-2つのサンプル間の平均差は0ではありません
    :param data:
    :return:
    """
    # data = pd.DataFrame(data)
    # data1 = np.array(data[target_1])
    # data2 = np.array(data[target_2])
    # test
    data1 = data
    data2 = data
    t_test, t_pval = stats.ttest_rel(data1, data2)
    if t_pval < 0.05:
        print(f"帰無仮説$H_{0}$を棄却 :  {round(t_pval, 4)} < 0.05")
    else:
        print(f"帰無仮説を棄却できません : {round(t_pval, 4)} > 0.05")


def z_test(data, h0=156):
    """
    次の場合はZ検定を使用します。
        ・サンプルサイズが30を超えています。
            それ以外の場合は、tテストを使用してください。
        ・データポイントは互いに独立している必要があります。
            つまり、あるデータポイントは関連していないか、別のデータポイントに影響を与えません。
        ・データは正規分布する必要があります。
            ただし、サンプルサイズが大きい（30を超える）場合、これは必ずしも重要ではありません。
        ・データは母集団からランダムに選択する必要があります。
            各アイテムが選択される可能性は同じです。
        ・サンプルサイズは、可能な限り等しくする必要があります。
    ----------------------------------------------------
    (例)血圧のz検定を使用しており、平均値は156の1サンプルZ検定
    H_{0} : 平均値である
    H_{1} : 平均値ではない
    :param data:
    :param h0:
    :return:
    """
    z_test, z_pval = statsmd.ztest(data, x2=None, value=h0)
    if z_pval < 0.05:
        print(f"帰無仮説$H_{0}$を棄却 :  {round(z_pval, 4)} < 0.05")
    else:
        print(f"帰無仮説を棄却できません : {round(z_pval, 4)} > 0.05")


def two_z_test(data, target_1='', target_2='', h0=0):
    """
    2標本Z検定-2標本z検定では、ここでのt検定と同様に、2つの独立したデータグループをチェックし、
    2つのグループの標本平均が等しいかどうかを判断します。
    ----------------------------------------------
    (例)
    H0：2つのグループの平均は0です
    H1：2つのグループの平均が0ではない
    ----------------------------------------------
    :param data:
    :param target_1:
    :param target_2:
    :param h0:
    :return:
    """
    data1 = data[target_1]
    data2 = data[target_2]
    z_test, z_pval = statsmd.ztest(x1=data1, x2=data2, value=h0, alternative='two-sided')
    if z_pval < 0.05:
        print(f"帰無仮説$H_{0}$を棄却 :  {round(z_pval, 4)} < 0.05")
    else:
        print(f"帰無仮説を棄却できません : {round(z_pval, 4)} > 0.05")


def anova_one_way_f_test(data, target1='', target2='', target3=''):
    """
    ANOVA（F-TEST）：- 2つのグループを処理する場合、t検定はうまく機能しますが、
    3つ以上のグループを同時に比較したい場合があります。
    たとえば、投票者の年齢が人種などのカテゴリ変数に基づいて異なるかどうかをテストする場合は、
    各レベルの平均を比較するか、変数をグループ化する必要があります。
    グループのペアごとに個別のt検定を実行することもできますが、多くのテストを実行すると、
    誤検知の可能性が高くなります。分散分析またはANOVAは、複数のグループを同時に比較できる
    統計的推論テストです。
    -----------------------------------------------------------------
    z分布やt分布とは異なり、F分布には負の値はありません。これは、各偏差を2乗するため、
    グループ間およびグループ内の変動が常に正であるためです。

    一元配置F検定（Anova）：-平均類似度とFスコアに基づいて、2つ以上のグループが類似している
    かどうかを判断します。
    ------------------------------------------------------------------
    (例)　植物には3つの異なるカテゴリとその重量があり、
    3つのグループすべてが類似しているかどうかを確認する必要があります
    :param data:
    :param target1:
    :param target2:
    :param target3:
    :return:
    """
    data1 = data[target1]
    data2 = data[target2]
    data3 = data[target3]
    F, p = stats.f_oneway(data1, data2, data3)
    print(f'pの有意性は次の通り : {p}')
    if p < 0.05:
        print(f"帰無仮説$H_{0}$を棄却 :  {round(p, 4)} < 0.05")
    else:
        print(f"帰無仮説を棄却できません : {round(p, 4)} > 0.05")


def anova_two_way_f_test(data, ols_1='Yield', ols_2='Fert', ols_3='Water'):
    """
    双方向F検定： -2方向F検定は、1方向f検定の拡張であり、2つの独立変数と2+グループがある場合に使用。
    2方向F検定では、どの変数が優勢であるかはわかりません。
    個々の有意性を確認する必要がある場合は、事後テストを実行する必要があります。
    -----------------------------------------------------------------
    (例)
    総平均収穫量（サブグループではない平均収穫量）、
    および各要因別、およびグループ化された要因別の平均収穫量を見る
    :param data:
    :param ols_1:
    :param ols_2:
    :param ols_3:
    :return:
    """
    model = ols(formula=f'{ols_1} ~ C({ols_2}) * C({ols_3})', data=data).fit()
    print(f"Overall model F({model.df_model: .0f}, {model.df_resid: .0f})={model.fvalue: .3f}, p={model.f_pvalue: .4f}")
    res = sm.stats.anova_lm(model, typ=2)
    print(res)


def chi_square_test(df_chi):
    """
    <改修予定：テスト段階なので指定データ形式にのみしか使用不能>
    ------------------------------------------------------
    カイ二乗検定-この検定は、単一の母集団から2つのカテゴリ変数がある場合に適用されます。
    これは、2つの変数の間に有意な関連があるかどうかを判断するために使用されます
    ------------------------------------------------------
    (例)
    選挙調査では、有権者は性別（男性または女性）と投票の好み（民主党、共和党、または独立）
    によって分類される場合があります。性別が投票の好みに関連しているかどうかを判断するために、
    独立性のカイ2乗検定を使用できます。
    :param df_chi:
    :return:
    """
    contingency_table = pd.crosstab(df_chi["Gender"], df_chi["Shopping?"])
    print('contingency_table :-\n', contingency_table)

    Observed_Values = contingency_table.values
    print("Observed Values :-\n", Observed_Values)

    b = stats.chi2_contingency(contingency_table)
    Expected_Values = b[3]
    print("Expected Values :-\n", Expected_Values)

    no_of_rows = len(contingency_table.iloc[0:2, 0])
    no_of_columns = len(contingency_table.iloc[0, 0:2])
    ddof = (no_of_rows - 1) * (no_of_columns - 1)
    print("Degree of Freedom:-", ddof)
    alpha = 0.05
    chi_square = sum([(o - e) ** 2. / e for o, e in zip(Observed_Values, Expected_Values)])
    chi_square_statistic = chi_square[0] + chi_square[1]
    print("chi-square statistic:-", chi_square_statistic)
    critical_value = chi2.ppf(q=1 - alpha, df=ddof)
    print('critical_value:', critical_value)

    # p-value
    p_value = 1 - chi2.cdf(x=chi_square_statistic, df=ddof)
    print('p-value:', p_value)
    print('Significance level: ', alpha)
    print('Degree of Freedom: ', ddof)
    print('chi-square statistic:', chi_square_statistic)
    print('critical_value:', critical_value)
    print('p-value:', p_value)
    if chi_square_statistic >= critical_value:
        print("Reject H0,There is a relationship between 2 categorical variables")
    else:
        print("Retain H0,There is no relationship between 2 categorical variables")

    if p_value <= alpha:
        print("Reject H0,There is a relationship between 2 categorical variables")
    else:
        print("Retain H0,There is no relationship between 2 categorical variables")

if __name__ == "__main__":

    ages = [32, 34, 29, 29, 22, 39, 39, 37, 38, 36, 30, 26, 22, 22]
    one_sample_t_test(ages)

    two_sampled_t_test(ages)

    paired_t_test(ages)

    df_anova2 = pd.read_csv(
        "https://raw.githubusercontent.com/Opensourcefordatascience/Data-sets/master/crop_yield.csv")
    anova_two_way_f_test(df_anova2)

    chi_test = pd.DataFrame({'Gender': ['Male', 'Female', 'Male', 'Female',
                                        'Female', 'Male', 'Male', 'Female', 'Female'],
                             'Shopping?': ['No', 'Yes', 'Yes', 'Yes',
                                           'Yes', 'Yes', 'No', 'No', 'No']})
    chi_square_test(chi_test)