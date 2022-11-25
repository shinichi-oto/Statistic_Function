import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import clc


def xmrchart(data, n=2):
    """
    連続データの管理図
    -------------------------------------------------------------------------------
    xチャートとmRチャート
    xチャート（個別チャート）とmRチャートは、特定の時間に取得された個々のサンプルに基づいて、
    プロセスの平均と変動を監視するために使用されます。

    mRチャートをxチャートと一緒に使用するには、サンプルサイズnが1に等しくなければなりません。
    ・xチャート:  y軸は平均と管理限界
     　          x軸はサンプル単位
    ・mRチャート: y軸は移動範囲の総平均と管理限界
       　        x軸はサンプル単位
    -------------------------------------------------------------------------------
    両管理限界線は一般的に標準偏差の三倍の３σ
    中心線(CL)　: 平均値
    上方管理限界線(UCL : Upper Control limit) : 平均値+3 * 標準偏差(σ)
    下方管理限界線(LCL : Lower Control limit) : 平均値-3 * 標準偏差(σ)

    :param data:
    :param n:
    :return:
    """
    clcd2 = clc.cLC()
    clcd3 = clc.d34N()
    # c.query('SSize == @n')
    d2 = float(clcd2.query(f'SSize == {n}')['d2'])
    d3 = float(clcd3.query(f'N == {n}')['d3'])

    xi = pd.Series(data)
    MR = [np.nan]

    i = 1
    for mr in range(1, len(xi)):
        MR.append(abs(xi[i] - xi[i-1]))
        i += 1

    MR = pd.Series(MR)

    cdata = pd.concat([x, MR], axis=1).rename(columns={0: "x", 1: "mR"})

    fig, ax = plt.subplots(2, figsize=(15, 15))  # sharex=True

    # X - Chart
    ax[0].plot(cdata['x'], linestyle='-', marker='o', color='black')
    ax[0].axhline(np.mean(cdata['x']), color='blue')
    ax[0].axhline(np.mean(cdata['x']) + 3 * np.mean(cdata['mR'][1:len(cdata['mR'])]) / d2, color='red',
                   linestyle='dashed')
    ax[0].axhline(np.mean(cdata['x']) - 3 * np.mean(cdata['mR'][1:len(cdata['mR'])]) / d2, color='red',
                   linestyle='dashed')
    ax[0].set_title('Individual Chart')
    ax[0].set(xlabel='Unit', ylabel='Value')

    # mR - Chart
    ax[1].plot(cdata['mR'], linestyle='-', marker='o', color='black')
    ax[1].axhline(np.mean(cdata['mR'][1:len(cdata['mR'])]), color='blue')
    ax[1].axhline(
        np.mean(cdata['mR'][1:len(cdata['mR'])]) + 3 * np.mean(cdata['mR'][1:len(cdata['mR'])]) * d3,
        color='red', linestyle='dashed')
    ax[1].axhline(
        np.mean(cdata['mR'][1:len(cdata['mR'])]) - 3 * np.mean(cdata['mR'][1:len(cdata['mR'])]) * d3,
        color='red', linestyle='dashed')
    ax[1].set_ylim(bottom=0)
    ax[1].set_title('mR Chart')
    ax[1].set(xlabel='Unit', ylabel='Range')

    # X線グラフの管理限界から外れたポイントの検証
    i = 0
    ocx = []
    control = True
    for unit in cdata['x']:
        if unit > np.mean(cdata['x']) + 3 * np.mean(
                cdata['mR'][1:len(cdata['mR'])]) / d2 or unit < np.mean(cdata['x']) - 3 *\
                np.mean(cdata['mR'][1:len(cdata['mR'])]) / d2:
            ocx.append(unit)
            print(f'X-Chart : Unit Number: {i} is out of control limits, Limit Value {unit}')
            control = False
        i += 1
    if control is True:
        print('All points within control limits.')

    # mRチャートのコントロールリミットを逸脱したポイントの検証
    i = 0
    ocmr = []
    control = True
    for unit in cdata['mR']:
        if unit > np.mean(cdata['mR'][1:len(cdata['mR'])]) + 3 * np.mean(
                cdata['mR'][1:len(cdata['mR'])]) * d3 or unit < np.mean(cdata['mR'][1:len(cdata['mR'])]) - 3 *\
                np.mean(cdata['mR'][1:len(cdata['mR'])]) * d3:
            ocmr.append(unit)
            print(f'mR-Chart : Unit Number: {i} is out of control limits, Unit Value {unit}')
            control = False
        i += 1
    if control is True:
        print('All points within control limits.')

    plt.show()


if __name__ == '__main__':

    np.random.seed(43)
    x = pd.Series(np.random.normal(loc=10, scale=2, size=1000))
    xmrchart(x)

    # help(xmrchart)
