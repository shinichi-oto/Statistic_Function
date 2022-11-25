import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import clc


def xrchart(data):
    """
    XチャートとRチャート
    -------------------------------------------------------------------
    xチャートとRチャートは、特定の時間に採取されたサンプルに基づいて、
    プロセスの平均と変動を監視するために使用されます。

    Rチャートをxバーチャートと一緒に使用するには、サンプルサイズnが1より大きく11未満である必要があります。(1<n<11)
    xチャート:      y軸は総平均と管理限界
                   x軸はサンプルグループ
    Rチャート :     y軸は範囲の総平均と管理限界
                   x軸はサンプルグループ
    -------------------------------------------------------------------
    :input: np.array
    :param data:
    :return:
    """
    df = pd.DataFrame(data)
    n = len(df.columns)
    if not 1 < n < 11:
        print(f"Not XR-Chart (1 < {n} < 11)")
        if n > 11:
            print(f"XS-Chart ({n} > 11)")
    df['mean'] = df.mean(axis=1)
    df['r'] = df.max(axis=1) - df.min(axis=1)

    clc1 = clc.cLC()
    A2 = float(clc1.query(f'SSize == {n}')['A2'])
    D3 = float(clc1.query(f'SSize == {n}')['D3'])
    D4 = float(clc1.query(f'SSize == {n}')['D4'])

    bar = df['mean']
    r = df['r']

    fig, ax = plt.subplots(2, figsize=(15, 15))

    # X chart
    ax[0].plot(bar, linestyle='-', marker='o', color='black')
    ax[0].axhline((np.mean(bar)), color='blue')
    ax[0].axhline((np.mean(bar) + A2 * np.mean(r)), color='red', linestyle='dashed')
    ax[0].axhline((np.mean(bar) - A2 * np.mean(r)), color='red', linestyle='dashed')
    ax[0].set_title('X - Chart')
    ax[0].set(xlabel='Index', ylabel='Mean')

    # R chart
    ax[1].plot(r, linestyle='-', marker='o', color='black')
    ax[1].axhline((np.mean(r)), color='blue')
    ax[1].axhline((D4 * np.mean(r)), color='red', linestyle='dashed')
    ax[1].axhline((D3 * np.mean(r)), color='red', linestyle='dashed')
    ax[1].set_ylim(bottom=0)
    ax[1].set_title('R - Chart')
    ax[1].set(xlabel='Index', ylabel='Range')

    i = 0
    control = True
    for xi in bar:
        if xi > np.mean(bar) + A2 * np.mean(r) or xi < np.mean(bar) - A2 * np.mean(r):
            print(f'X-Unit {i} out of mean control limits! {x}')
            control = False
        i += 1
    if control is True:
        print('X-Chart : All points within control limits.')

    # Validate points out of control limits for R chart
    i = 0
    control = True
    for rx in r:
        if rx > D4 * np.mean(r) or rx < D3 * np.mean(r):
            print(f'R-Unit {i} out of range control limits! {rx}')
            control = False
        i += 1
    if control is True:
        print('R-Chart All points within control limits.')

    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    # Create dummy data 10*5 = n=5
    x = np.array([list(np.random.normal(loc=10, scale=2, size=5)),
                  list(np.random.normal(loc=10, scale=2, size=5)),
                  list(np.random.normal(loc=10, scale=2, size=5)),
                  list(np.random.normal(loc=10, scale=2, size=5)),
                  list(np.random.normal(loc=10, scale=2, size=5)),
                  list(np.random.normal(loc=17, scale=2, size=5)),
                  list(np.random.normal(loc=10, scale=2, size=5)),
                  list(np.random.normal(loc=10, scale=2, size=5)),
                  list(np.random.normal(loc=10, scale=2, size=5)),
                  list(np.random.normal(loc=10, scale=2, size=5))])

    xrchart(x)