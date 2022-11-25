import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cupnpchart(data, chart="c", target="defects", target_group="group_size"):
    """
    離散データの管理図
    ------------------------------------------
    cチャート
    cチャート :   サイズnの固定サンプルの欠陥の総数を監視するために使用。
                 y軸はサンプルごとの不適合の数、
                 x軸はサンプルグループ
                 ------------------------------------------------------
                 CL = Σc/n
                      Σc : 不適合総和
                      n  : サンプル数
                 UCL = \={c} + 3√\={c}
                 LCL = \={c} - 3√\={c}
    ------------------------------------------
    uチャート
    uチャート :  サイズnのさまざまなサンプルのユニットあたりの欠陥の総数を監視するために使用
                ユニットに複数の欠陥がある可能性があることを前提
                y軸は単一ユニットあたりの欠陥数
                x軸はサンプルグループ
                -------------------------------------------------------
                CL \={u} = Σc/Σn
                     Σc : 欠陥数の総和
                     Σn : サンプル数総和
                UCL = \={u} + 3 √\={u}/n
                LCL = \={u} - 3 √\={u}/n
    ------------------------------------------
    pチャート
    pチャート :  サイズnのさまざまなサンプルの不適合ユニットの割合を監視するために使用
                これは、各ユニットに2つの可能性（つまり、欠陥があるかどうか）しかない二項分布に基く
                y軸は不適合ユニットの割合
                x軸はサンプルグループ
                -------------------------------------------------------
                p=pn/n
                    pn : サンプルの不良個数
                    n  : サンプルサイズ
                CL \={p} = Σpn/Σn
                UCL = \={p} + 3√\={p}(1-\={p})/n
                LCL = \={p} - 3√\={p}(1-\={p})/n

    ------------------------------------------
    np-チャート
    npチャート : サイズnの固定サンプルの不適合ユニットの数を監視するために使用
                y軸は不適合ユニットの総数
                x軸はサンプルグループ
                -------------------------------------------------------
                \={p} = Σpn/Σn = Σpn/kn
                CL　\={p}n=Σpn/k
                    pn  : 各sampleの不良個数
                    Σpn : 不適合ユニット個数の総和
                    k   :  sample数
                UCL \={p}n+3√\={p}n(1-\={p})
                LCL \={p}n+3√\={p}n(1-\={p})

    :param data:
    :param chart: c u p np
    :param target: str -> target-name
    :param target_group: str -> group-name
    :return:
    """
    df = pd.DataFrame(data)

    if chart == "c":
        # C chart
        fig, ax = plt.subplots(1, figsize=(15, 15))
        ax.plot(df[target], linestyle='-', marker='o', color='black')
        ax.axhline(np.mean(df[target]), color='blue')
        ax.axhline(np.mean(df[target]) + 3 * np.sqrt(np.mean(df[target])), color='red', linestyle='dashed')
        ax.axhline(np.mean(df[target]) - 3 * np.sqrt(np.mean(df[target])), color='red', linestyle='dashed')
        ax.set_ylim(bottom=0)
        ax.set_title('C - Chart')
        ax.set(xlabel='Index', ylabel=f'{target} Count')

        i = 0

        control = True
        for x in df[target]:
            if x > np.mean(df[target]) + 3 * np.sqrt(np.mean(df[target])) or\
                    x < np.mean(df[target]) - 3 * np.sqrt(np.mean(df[target])):
                print(f'C {i} out of mean control limits! {x}')
                control = False
            i += 1
        if control is True:
            print('C-Chart : All points within control limits.')

    elif chart == "u":
        # U chart
        df["u"] = df[target] / df[target_group]

        plt.figure(figsize=(15, 15))
        plt.plot(df["u"], linestyle='-', marker='o', color='black')
        plt.axhline(df["u"].mean(), color='blue')
        plt.step(x=range(0, len(df["u"])), y=df["u"].mean() + 3 * np.sqrt(df["u"].mean() / df[target_group]),
                 color='red', linestyle='dashed')
        plt.step(x=range(0, len(df["u"])), y=df["u"].mean() - 3 * np.sqrt(df["u"].mean() / df[target_group]),
                 color='red', linestyle='dashed')
        plt.ylim(bottom=0)
        plt.title('U - Chart')
        plt.xlabel(f'{target_group} Group')
        plt.ylabel('Fraction Defective')

        i = 0

        control = True
        for x in df["u"]:
            if x > df["u"].mean() + 3 * np.sqrt(df["u"].mean() / df[target_group][i]) or\
                    x < df["u"].mean() - 3 * np.sqrt(df["u"].mean() / df[target_group][i]):
                print(f'U {i} out of fraction defective control limits! {x}')
                control = False
            i += 1
        if control is True:
            print('U-Chart : All points within control limits.')

    elif chart == "p":
        df["p"] = df[target] / df[target_group]

        plt.figure(figsize=(15, 15))
        plt.plot(df["p"], linestyle='-', marker='o', color='black')
        plt.axhline(df["p"].mean(), color='blue')
        plt.step(x=range(0, len(df["p"])),
                 y=df["p"].mean() + 3 * (np.sqrt(df["p"].mean() * (1 - df['p'].mean()) / (df[target_group]))),
                 color='red', linestyle='dashed')
        plt.step(x=range(0, len(df["p"])),
                 y=df["p"].mean() - 3 * (np.sqrt(df["p"].mean() * (1 - df['p'].mean()) / df[target_group])),
                 color='red', linestyle='dashed')
        plt.ylim(bottom=0)
        plt.title('P - Chart')
        plt.xlabel(f'{target_group} Group')
        plt.ylabel('Fraction Defective')

        i = 0

        control = True
        for x in df["p"]:
            if x > (df["p"].mean()) + 3 * (np.sqrt(df["p"].mean()) * (1 - df['p'].mean()) /
                                           (df[target_group].mean())) or\
                    x < df["p"].mean() - 3 * (np.sqrt(df["p"].mean() * (1 - df['p'].mean()) /
                                                      df[target_group].mean())):
                print(f'P {i} out of fraction defective control limits! {x}')
                control = False
            i += 1
        if control is True:
            print('P-Chart : All points within control limits.')

    elif chart == "np":
        df["np"] = df[target] / df[target_group]

        plt.figure(figsize=(15, 7.5))
        plt.plot(df['np'], linestyle='-', marker='o', color='black')
        plt.axhline(df['np'].mean() + 3 * (np.sqrt(df['np'].mean() *
                                                   (1 - df['np'].mean())) / (df[target_group].mean())),
                    color='red', linestyle='dashed')
        plt.axhline(df['np'].mean() - 3 * (np.sqrt((df['np'].mean()) *
                                                   (1 - df['np'].mean())) / (df[target_group].mean())),
                    color='red', linestyle='dashed')
        plt.axhline(df['np'].mean(), color='blue')
        plt.ylim(bottom=0)
        plt.title('np Chart')
        plt.xlabel('Group')
        plt.ylabel('Fraction Defective')

        i = 0
        control = True
        for x in df['np']:
            if x > df['np'].mean() + 3 * (np.sqrt((df['np'].mean()) *
                                                   (1 - df['np'].mean())) / (df[target_group].mean())) or \
                    x < df['np'].mean() - 3 * (np.sqrt((df['np'].mean()) *
                                                   (1 - df['np'].mean())) / (df[target_group].mean())):
                print(f'np {i} out of fraction defective control limits! {x}')
                control = False
            i += 1
        if control is True:
            print('np-Chart : All points within control limits.')

    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    c = {'defects': np.random.randint(0, 5, 10).tolist(),
         'group_size': np.repeat(10, 10).tolist()}
    c = pd.DataFrame(c)

    u = {'defects': np.random.randint(1, 5, 10).tolist(),
         'group_size': np.random.randint(10, 15, 10).tolist()}
    u = pd.DataFrame(u)

    p = {'defects': np.random.randint(1, 5, 10).tolist(),
         'group_size': np.random.randint(10, 15, 10).tolist()}
    p = pd.DataFrame(p)

    cc = {'defects': np.random.randint(1, 5, 10).tolist(),
          'group_size': np.repeat(10, 10).tolist()}
    cc = pd.DataFrame(cc)

    cupnpchart(cc, chart="np")
