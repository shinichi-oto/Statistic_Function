import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import clc


def xschart(data):
    """
    XバーチャートとRチャート
    -------------------------------------------------------------------
    xバーとRチャートは、特定の時間に採取されたサンプルに基づいて、
    プロセスの平均と変動を監視するために使用

    Rチャートをxバーチャートと一緒に使用するには、サンプルサイズnが1より大きく11未満である必要があります。(1<n<11)
    xチャート:  y軸は総平均と管理限界
                   x軸はサンプルグループ
    Rチャート :     y軸は範囲の総平均と管理限界
                   x軸はサンプルグループ
    xバーチャートとsチャート
    -------------------------------------------------------------------
    同様に、xとsチャートは、特定の時間に取得されたサンプルに基づいて
    プロセスの平均と変動を監視するために使用

    sチャートをxチャートと一緒に使用するには、サンプルサイズnが10単位より大きくなければなりません。
    xチャート:  y軸は総平均と管理限界
            　 x軸はサンプルグループを示します。
    sチャート:  y軸は標準偏差の総平均と管理限界
               x軸はサンプルグループを示します。
    :input: np.array
    :param data:
    :return:
    """
    df = pd.DataFrame(data)
    n = len(df.columns)
    if not n > 10:
        print(f"Not XS-Chart ({n} > 10)")
        if n < 10:
            print(f"XR-Chart (1 < {n} < 10)")
    df['mean'] = df.mean(axis=1)
    df['s'] = df.std(axis=1)

    clc1 = clc.cLC()
    A3 = float(clc1.query(f'SSize == {n}')['A3'])
    B3 = float(clc1.query(f'SSize == {n}')['B3'])
    B4 = float(clc1.query(f'SSize == {n}')['B4'])

    bar = df['mean']
    s = df['s']

    fig, ax = plt.subplots(2, figsize=(15, 15))

    # X chart
    ax[0].plot(bar, linestyle='-', marker='o', color='black')
    ax[0].axhline((np.mean(bar)), color='blue')
    ax[0].axhline((np.mean(bar) + A3 * np.mean(s)), color='red', linestyle='dashed')
    ax[0].axhline((np.mean(bar) - A3 * np.mean(s)), color='red', linestyle='dashed')
    ax[0].set_title('X - Chart')
    ax[0].set(xlabel='Index', ylabel='Mean')

    # R chart
    ax[1].plot(s, linestyle='-', marker='o', color='black')
    ax[1].axhline((np.mean(s)), color='blue')
    ax[1].axhline((B4 * np.mean(s)), color='red', linestyle='dashed')
    ax[1].axhline((B3 * np.mean(s)), color='red', linestyle='dashed')
    ax[1].set_ylim(bottom=0)
    ax[1].set_title('S - Chart')
    ax[1].set(xlabel='Index', ylabel='std')

    i = 0
    control = True
    for xi in bar:
        if xi > np.mean(bar) + A3 * np.mean(s) or xi < np.mean(bar) - A3 * np.mean(s):
            print(f'X-Unit {i} out of mean control limits! {x}')
            control = False
        i += 1
    if control is True:
        print('X-Chart : All points within control limits.')

    # Validate points out of control limits for R chart
    i = 0
    control = True
    for sx in s:
        if sx > B4 * np.mean(s) or sx < B3 * np.mean(s):
            print(f'S-Unit {i} out of std control limits! {sx}')
            control = False
        i += 1
    if control is True:
        print('S-Chart All points within control limits.')

    plt.show()


if __name__ == "__main__":

    np.random.seed(42)
    x = np.array([list(np.random.normal(loc=10, scale=2, size=11)),
                  list(np.random.normal(loc=10, scale=2, size=11)),
                  list(np.random.normal(loc=10, scale=2, size=11)),
                  list(np.random.normal(loc=10, scale=2, size=11)),
                  list(np.random.normal(loc=10, scale=2, size=11)),
                  list(np.random.normal(loc=10, scale=2, size=11)),
                  list(np.random.normal(loc=10, scale=2, size=11)),
                  list(np.random.normal(loc=13, scale=2, size=11)),
                  list(np.random.normal(loc=10, scale=2, size=11)),
                  list(np.random.normal(loc=10, scale=2, size=11))])

    xschart(x)