# -*- coding: utf-8 -*-
"""
蚁群算法求解TSP问题
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
    for i in range(len(line) - 1):
        dis = dis_matrix.loc[line[i], line[i + 1]]  # 计算距离
        dis_sum = dis_sum + dis
    dis = dis_matrix.loc[line[-1], line[0]]
    dis_sum = dis_sum + dis

    return round(dis_sum, 1)


def intialize(CityCoordinates, antNum):
    """
    初始化，为蚂蚁分配初始城市
    输入：CityCoordinates-城市坐标;antNum-蚂蚁数量
    输出：cityList-蚂蚁初始城市列表，记录蚂蚁初始城市;cityTabu-蚂蚁城市禁忌列表，记录蚂蚁未经过城市
    """
    cityList, cityTabu = [None] * antNum, [None] * antNum  # 初始化
    for i in range(len(cityList)):
        city = random.randint(0, len(CityCoordinates) - 1)  # 初始城市，默认城市序号为0开始计算
        cityList[i] = [city]
        cityTabu[i] = list(range(len(CityCoordinates)))
        cityTabu[i].remove(city)

    return cityList, cityTabu


def select(antCityList, antCityTabu, trans_p):
    '''
    轮盘赌选择，根据出发城市选出途径所有城市
    输入：trans_p-概率矩阵;antCityTabu-城市禁忌表，即未经过城市;
    输出：完整城市路径-antCityList;
    '''
    while len(antCityTabu) > 0:
        if len(antCityTabu) == 1:
            nextCity = antCityTabu[0]
        else:
            fitness = []
            for i in antCityTabu: fitness.append(trans_p.loc[antCityList[-1], i])  # 取出antCityTabu对应的城市转移概率
            sumFitness = sum(fitness)
            randNum = random.uniform(0, sumFitness)
            accumulator = 0.0
            for i, ele in enumerate(fitness):
                accumulator += ele
                if accumulator >= randNum:
                    nextCity = antCityTabu[i]
                    break
        antCityList.append(nextCity)
        antCityTabu.remove(nextCity)

    return antCityList


def calTrans_p(pheromone, alpha, beta, dis_matrix, Q):
    '''
    根据信息素计算转移概率
    输入：pheromone-当前信息素；alpha-信息素重要程度因子；beta-启发函数重要程度因子；dis_matrix-城市间距离矩阵；Q-信息素常量；
    输出：当前信息素+增量-transProb
    '''
    transProb = Q / dis_matrix  # 初始化transProb存储转移概率，同时计算增量
    for i in range(len(transProb)):
        for j in range(len(transProb)):
            transProb.iloc[i, j] = pow(pheromone.iloc[i, j], alpha) * pow(transProb.iloc[i, j], beta)

    return transProb


def updatePheromone(pheromone, fit, antCity, rho, Q):
    '''
    更新信息素，蚁周算法
    输入：pheromone-当前信息素；fit-路径长度；antCity-路径；rho-ρ信息素挥发因子；Q-信息素常量
    输出：更新后信息素-pheromone
    '''
    for i in range(len(antCity) - 1):
        pheromone.iloc[antCity[i], antCity[i + 1]] += Q / fit
    pheromone.iloc[antCity[-1], antCity[0]] += Q / fit

    return pheromone


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
    iterMax = 100  # 迭代次数
    iterI = 1  # 当前迭代次数
    # ACO参数
    antNum = 50  # 蚂蚁数量
    alpha = 2  # 信息素重要程度因子
    beta = 1  # 启发函数重要程度因子
    rho = 0.2  # 信息素挥发因子
    Q = 100.0  # 常数

    best_fit = math.pow(10, 10)  # 较大的初始值，存储最优解
    best_line = []  # 存储最优路径

    # 随机生成城市数据,城市序号为0,1,2,3...
    # CityCoordinates = [(random.randint(MinCoordinate,MaxCoordinate),random.randint(MinCoordinate,MaxCoordinate)) for i in range(CityNum)]
    CityCoordinates = [(88, 16), (42, 76), (5, 76), (69, 13), (73, 56), (100, 100), (22, 92), (48, 74), (73, 46),
                       (39, 1), (51, 75), (92, 2), (101, 44), (55, 26), (71, 27), (42, 81), (51, 91), (89, 54),
                       (33, 18), (40, 78)]

    # 计算城市间距离,生成矩阵
    dis_matrix = pd.DataFrame(data=None, columns=range(len(CityCoordinates)), index=range(len(CityCoordinates)))
    for i in range(len(CityCoordinates)):
        xi, yi = CityCoordinates[i][0], CityCoordinates[i][1]
        for j in range(len(CityCoordinates)):
            xj, yj = CityCoordinates[j][0], CityCoordinates[j][1]
            if (xi == xj) & (yi == yj):
                dis_matrix.iloc[i, j] = round(math.pow(10, 10))
            else:
                dis_matrix.iloc[i, j] = round(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2), 2)

    pheromone = pd.DataFrame(data=Q, columns=range(len(CityCoordinates)),
                             index=range(len(CityCoordinates)))  # 初始化信息素，所有路径都为Q
    trans_p = calTrans_p(pheromone, alpha, beta, dis_matrix, Q)  # 计算初始转移概率

    while iterI <= iterMax:
        '''
        每一代更新一次环境因素导致的信息素减少，每一代中的每一个蚂蚁完成路径后，都进行信息素增量更新（采用蚁周模型）和转移概率更新；
        每一代开始都先初始化蚂蚁出发城市；
        '''
        antCityList, antCityTabu = intialize(CityCoordinates, antNum)  # 初始化城市
        fitList = [None] * antNum  # 适应值列表

        for i in range(antNum):  # 根据转移概率选择后续途径城市，并计算适应值
            antCityList[i] = select(antCityList[i], antCityTabu[i], trans_p)
            fitList[i] = calFitness(antCityList[i], dis_matrix)  # 适应度，即路径长度
            pheromone = updatePheromone(pheromone, fitList[i], antCityList[i], rho, Q)  # 更新当前蚂蚁信息素增量
            trans_p = calTrans_p(pheromone, alpha, beta, dis_matrix, Q)

        if best_fit >= min(fitList):
            best_fit = min(fitList)
            best_line = antCityList[fitList.index(min(fitList))]

        print(iterI, best_fit)  # 打印当前代数和最佳适应度值
        iterI += 1  # 迭代计数加一
        pheromone = pheromone * (1 - rho)  # 信息素挥发更新

    print(best_line)  # 路径顺序
    draw_path(best_line, CityCoordinates)  # 画路径图
