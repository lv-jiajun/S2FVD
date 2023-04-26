# -*- coding: utf-8 -*-
"""
模拟退火算法法求解TSP问题
随机在（0,101）二维平面生成20个点
距离最小化
"""
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文


# 计算路径距离，即评价函数
def calFitness(line, dis_matrix):
    dis_sum = 0
    dis = 0
    for i in range(len(line)):
        if i < len(line) - 1:
            dis = dis_matrix.loc[line[i], line[i + 1]]  # 计算距离
            dis_sum = dis_sum + dis
        else:
            dis = dis_matrix.loc[line[i], line[0]]
            dis_sum = dis_sum + dis
    return round(dis_sum, 1)


def traversal_search(line, dis_matrix):
    # 随机交换生成100个个体，选择其中表现最好的返回
    i = 0  # 生成个体计数
    line_value, line_list = [], []
    while i <= 100:
        new_line = line.copy()  # 复制当前路径
        exchange_max = random.randint(1, 5)  # 随机生成交换次数,城市数量较多时增加随机数上限效果较好
        exchange_time = 0  # 当前交换次数
        while exchange_time <= exchange_max:
            pos1, pos2 = random.randint(0, len(line) - 1), random.randint(0, len(line) - 1)  # 交换点
            new_line[pos1], new_line[pos2] = new_line[pos2], new_line[pos1]  # 交换生成新路径
            exchange_time += 1  # 更新交换次数

        new_value = calFitness(new_line, dis_matrix)  # 当前路径距离
        line_list.append(new_line)
        line_value.append(new_value)
        i += 1

    return min(line_value), line_list[line_value.index(min(line_value))]  # 返回表现最好的个体


def greedy(CityCoordinates, dis_matrix):
    '''贪婪策略构造初始解'''
    # 转换格式—dis_matrix
    dis_matrix = dis_matrix.astype('float64')
    for i in range(len(CityCoordinates)): dis_matrix.loc[i, i] = math.pow(10, 10)
    line = []  # 初始化
    now_city = random.randint(0, len(CityCoordinates) - 1)  # 随机生成出发城市
    line.append(now_city)  # 添加当前城市到路径
    dis_matrix.loc[:, now_city] = math.pow(10, 10)  # 更新距离矩阵，已经过城市不再被取出
    for i in range(len(CityCoordinates) - 1):
        next_city = dis_matrix.loc[now_city, :].idxmin()  # 距离最近的城市
        line.append(next_city)  # 添加进路径
        dis_matrix.loc[:, next_city] = math.pow(10, 10)  # 更新距离矩阵
        now_city = next_city  # 更新当前城市

    return line


# 画路径图
def draw_path(line, CityCoordinates):
    x, y = [], []
    for i in line:
        Coordinate = CityCoordinates[i]
        x.append(Coordinate[0])
        y.append(Coordinate[1])
    x.append(x[0])
    y.append(y[0])

    plt.plot(x, y, 'r-', color='#4169E1', alpha=0.8, linewidth=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


if __name__ == '__main__':
    # 参数
    CityNum = 20  # 城市数量
    MinCoordinate = 0  # 二维坐标最小值
    MaxCoordinate = 101  # 二维坐标最大值

    # SA参数
    Tend = 0.1  # 终止温度
    T = 100  # 初温
    beta = 0.99  # 退火步长

    best_value = math.pow(10, 10)  # 较大的初始值，存储最优解
    best_line = []  # 存储最优路径

    # 随机生成城市数据,城市序号为0,1，2,3...
    # CityCoordinates = [(random.randint(MinCoordinate,MaxCoordinate),random.randint(MinCoordinate,MaxCoordinate)) for i in range(CityNum)]
    CityCoordinates = [(88, 16), (42, 76), (5, 76), (69, 13), (73, 56), (100, 100), (22, 92), (48, 74), (73, 46),
                       (39, 1), (51, 75), (92, 2), (101, 44), (55, 26), (71, 27), (42, 81), (51, 91), (89, 54),
                       (33, 18), (40, 78)]

    # 计算城市之间的距离
    dis_matrix = pd.DataFrame(data=None, columns=range(len(CityCoordinates)), index=range(len(CityCoordinates)))
    for i in range(len(CityCoordinates)):
        xi, yi = CityCoordinates[i][0], CityCoordinates[i][1]
        for j in range(len(CityCoordinates)):
            xj, yj = CityCoordinates[j][0], CityCoordinates[j][1]
            dis_matrix.iloc[i, j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 2)

    # 随机构造初始解
    # line = list(range(len(CityCoordinates)));random.shuffle(line)
    # value = calFitness(line,dis_matrix)#初始路径距离

    # 贪婪构造初始解
    line = greedy(CityCoordinates, dis_matrix)
    value = calFitness(line, dis_matrix)  # 初始路径距离

    # 存储当前最优
    best_value, best_line = value, line
    draw_path(best_line, CityCoordinates)  # 初始路径

    while T >= Tend:
        new_value, new_line = traversal_search(line, dis_matrix)
        # print(random.random(),math.exp(-(new_value-value)/T))

        if new_value <= best_value:  # 优于最优解
            best_value, best_line = new_value, new_line  # 更新最优解
            line, value = new_line, new_value  # 更新当前解
        elif random.random() < math.exp(-(new_value - value) / T):
            line, value = new_line, new_value  # 更新当前解
        print('当前最优值 %.1f' % (best_value))
        T *= beta

    # 路径顺序
    print(best_line)
    # 画图
    draw_path(best_line, CityCoordinates)
