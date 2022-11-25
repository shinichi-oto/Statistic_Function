import pandas as pd


def cLC():
    return pd.DataFrame({"SSize": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                   22, 23, 24, 25],
                         "A2": [1.880, 1.023, 0.729, 0.577, 0.483, 0.419, 0.373, 0.337, 0.308, 0.285, 0.266,
                                0.248, 0.235, 0.223, 0.212, 0.203, 0.194, 0.187, 0.180, 0.173, 0.167, 0.162,
                                0.157, 0.153],
                         "A3": [2.659, 1.954, 1.628, 1.427, 1.287, 1.182, 1.099, 1.032, 0.975, 0.927, 0.886,
                                0.850, 0.817, 0.789, 0.763, 0.739, 0.718, 0.698, 0.680, 0.663, 0.647, 0.633,
                                0.619, 0.606],
                         "d2": [1.128, 1.693, 2.059, 2.326, 2.534, 2.704, 2.847, 2.970, 3.078, 3.173, 3.258,
                                3.336, 3.407, 3.472, 3.532, 3.588, 3.640, 3.689, 3.735, 3.778, 3.819, 3.858,
                                3.895, 3.931],
                         "D3": [0, 0, 0, 0, 0, 0.076, 0.136, 0.184, 0.223, 0.256, 0.283, 0.307, 0.328, 0.347,
                                0.363, 0.378, 0.391, 0.403, 0.415, 0.425, 0.434, 0.443, 0.451, 0.459],
                         "D4": [3.267, 2.574, 2.282, 2.114, 2.004, 1.924, 1.864, 1.816, 1.777, 1.744, 1.717,
                                1.693, 1.672, 1.653, 1.637, 1.622, 1.608, 1.597, 1.585, 1.575, 1.566, 1.557,
                                1.548, 1.541],
                         "B3": [0, 0, 0, 0, 0.030, 0.118, 0.185, 0.239, 0.284, 0.321, 0.354, 0.382, 0.406,
                                0.428, 0.448, 0.466, 0.482, 0.497, 0.510, 0.523, 0.534, 0.545, 0.555,
                                0.565],
                         "B4": [3.267, 2.568, 2.266, 2.089, 1.970, 1.882, 1.815, 1.761, 1.716, 1.679, 1.646,
                                1.618, 1.594, 1.572, 1.552, 1.534, 1.518, 1.503, 1.490, 1.477, 1.466, 1.455,
                                1.445, 1.435]}).set_index("SSize")


def d34N():
    return pd.DataFrame({"N": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                               21, 22, 23, 24, 25],
                         "d3": [0.8525, 0.8884, 0.8798, 0.8641, 0.848, 0.8332, 0.8198, 0.8078, 0.7971, 0.7873,
                                0.7785, 0.7704, 0.763, 0.7562, 0.7499, 0.7441, 0.7386, 0.7335, 0.7287, 0.7242,
                                0.7199, 0.7159, 0.7121, 0.7084],
                         "d4": [0.954, 1.588, 1.978, 2.257, 2.472, 2.645, 2.791, 2.915, 3.024, 3.121, 3.207,
                                3.285, 3.356, 3.422, 3.482, 3.538, 3.591, 3.64, 3.686, 3.73, 3.771, 3.811,
                                3.847, 3.883]}).set_index("N")


def d2N():
    return pd.DataFrame({"N": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                               21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
                               38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
                         "d2": [1.128, 1.693, 2.059, 2.326, 2.534, 2.704, 2.847, 2.97, 3.078, 3.173, 3.258,
                                3.336, 3.407, 3.472, 3.532, 3.588, 3.64, 3.689, 3.735, 3.778, 3.819, 3.858,
                                3.895, 3.931, 3.964, 3.997, 4.027, 4.057, 4.086, 4.113, 4.139, 4.165, 4.189,
                                4.213, 4.236, 4.259, 4.28, 4.301, 4.322, 4.341, 4.361, 4.379, 4.398, 4.415,
                                4.433, 4.45, 4.466, 4.482, 4.498]}).set_index("N")


if __name__ == "__main__":
    a = cLC()
    n = 5
    c = float(a.query('SSize == @n')['A2'])
    print(a)
    print(c)

    dn = d2N()
    print(dn)
