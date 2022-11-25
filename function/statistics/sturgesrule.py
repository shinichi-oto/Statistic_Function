import numpy as np


def sturgesrule(df, width=None):
    """
    スタージェスの公式
        n = 1+log10(N)/log10(2) == n = log2(N) + 1
    nを階級数とする
        範囲Rを階級数ｎで当分する
    :param width:
    :param n:
    :return:
    """
    st = round(1 + np.log10(len(df)) / np.log10(2))
    if width is None:
        wd = round((max(df) - min(df)) / st)
    else:
        wd = width
    return st, wd


if __name__ == '__main__':

    print(sturgesrule([1, 1, 1, 1, 110, 10, 120, 2223, 98]))