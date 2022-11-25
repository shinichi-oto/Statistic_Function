import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def paretochart(data, bar="bar_target", group_target='', line_size=4, hline=80, cat=2, rotation=45, group=False):
    """
    ■ 発生件数や発生金額での分類としては、
    ・不良発生件数や不良発生金額の分析
    ・クレーム件数やクレーム金額の分析
    ・設備の故障発生件数や故障発生金額の分析
    ・製品別の在庫個数や在庫金額の分析

    ■ 原因別での分類としては、
    ・不良発生原因別の分析
    ・クレーム発生原因別の分析
    ・故障発生原因別の分析
    ・在庫発生原因別の分析
    ---------------------------------------------------
    DataFrameをGroup化する場合、対象の一意な値に対してGroup化したいColumnsをgroupbyで収束する。
    ---------------------------------------------------
    パレート図の使用方法：line-plot : 累積
                        bar-plot : 対象のcount
    :param cat: 指定Category
    :param hline: line - %指定
    :param rotation: index rotation
    :param data: DataFrame
    :param bar: target_bar
    :param group_target: group化target
    :param line_size: line_size
    :param group: DataFrameのグループ化を使用するかどうか
    :return:
    """
    df = pd.DataFrame(data)
    if group is True:
        df = df.groupby(group_target).mean()
        df = df[[bar]]
    df.sort_values(by=bar, ascending=False, inplace=True)

    df['cumperc'] = df[bar].cumsum() / df[bar].sum() * 100

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(df.index, df[bar], color='steelblue', label='count')
    ax.axvline(x=cat, linestyle='--')
    ax.set_title('Pareto-Chart', size=30)
    ax.set(xlabel='Category', ylabel='Category-Count')

    ax1 = ax.twinx()
    ax1.plot(df.index, df['cumperc'], color='red', ms=line_size, marker='D', label='accumulate')
    ax1.yaxis.set_major_formatter(PercentFormatter())
    ax1.axhline(y=hline, linestyle='-.')
    ax1.set(ylabel='Cumsum-%')

    ax.tick_params(axis='y', colors='steelblue')
    ax1.tick_params(axis='y', colors='red')
    fig.legend(ncol=2, loc='lower center', frameon=False)

    for tick in ax.get_xticklabels():
        tick.set_rotation(rotation)

    plt.show()


import plotly.graph_objects as go


def plotly_pareto_chart(collection):
    collection = pd.Series(collection)
    counts = (collection.value_counts().to_frame('counts')
              .join(collection.value_counts(normalize=True).cumsum().to_frame('ratio')))

    fig = go.Figure([go.Bar(x=counts.index, y=counts['counts'], yaxis='y1', name='count'),
                     go.Scatter(x=counts.index, y=counts['ratio'], yaxis='y2', name='cumulative ratio',
                                hovertemplate='%{y:.1%}', marker={'color': '#000000'})])

    fig.update_layout(template='plotly_white', showlegend=False, hovermode='x', bargap=.3,
                      title={'text': 'Pareto Chart', 'x': .5},
                      yaxis={'title': 'count'},
                      yaxis2={'rangemode': "tozero", 'overlaying': 'y',
                              'position': 1, 'side': 'right',
                              'title': 'ratio',
                              'tickvals': np.arange(0, 1.1, .2),
                              'tickmode': 'array',
                              'ticktext': [str(i) + '%' for i in range(0, 101, 20)]})

    fig.show()


if __name__ == '__main__':
    df_test = pd.DataFrame({'count': [97, 140, 58, 6, 17, 32]})
    df_test.index = ['B', 'A', 'C', 'F', 'E', 'D']  # set_index()

    print(df_test)
    paretochart(df_test, bar='count', hline=80)

    # ------------------------------------------------------------------------------------
    data_a = np.random.choice(['USA', 'Canada', 'Russia', 'UK', 'Belgium',
                             'Mexico', 'Germany', 'Denmark'], size=500,
                            p=[0.43, 0.14, 0.23, 0.07, 0.04, 0.01, 0.03, 0.05])

    collection_a = pd.Series(data_a)
    print(collection_a)
    counts_a = (collection_a.value_counts().to_frame('counts')
              .join(collection_a.value_counts(normalize=True).cumsum().to_frame('ratio')))
    print(counts_a)
    # plotly_pareto_chart(data)
