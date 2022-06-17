#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/14 12:49
# @Author  : lanlin
# reference: http://t.csdn.cn/mORef


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


PI = 3.1415
DNA_SIZE = 24  # 二进制编码位数
POP_SIZE = 1000  # 种群中个体数量
CROSSOVER_RATE = 0.6  # 交叉概率
MUTATION_RATE = 0.001  # 变异概率
N_GENERATIONS = 100
X_BOUND = [-1.000, 1.000]
Y_BOUND = [-1.000, 1.000]


def F(x, y):
    """
    :param x: 染色体上的第一个基因，也就是x
    :param y: 染色体上的第二个基因，也就是y
    :return:  返回适配值。
              对于求最大值的情况，直接返回函数值；
              对于求最小值的情况，返回一个较大常数与函数值的差；
    """
    f = y**2.0 + x**2.0 - 0.3*np.cos(3*PI*x) - 0.4*np.cos(4*PI*y) + 0.7
    return 10-f  # 优化最小值


def plot_3d(ax):
    X = np.linspace(*X_BOUND, 100)
    Y = np.linspace(*Y_BOUND, 100)
    X, Y = np.meshgrid(X, Y)
    Z = F(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.pause(3)
    plt.show()


def get_fitness(pop):
    x, y = translateDNA(pop)
    pred = F(x, y)
    return pred


def translateDNA(pop):
    """
    :param pop: 表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目
    :return: 返回十进制的函数定义域数值
    """
    x_pop = pop[:, 0:DNA_SIZE]  # 前DNA_SIZE位表示X
    y_pop = pop[:, DNA_SIZE:]  # 后DNA_SIZE位表示Y

    # 二进制转十进制并压缩到取值范围
    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]
    return x, y


def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)

    return new_pop


def mutation(child, MUTATION_RATE=0.003):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE*2)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转


def select(pop, fitness):  # nature selection wrt pop's fitness
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=(fitness) / (fitness.sum()))
    return pop[idx]


def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:", fitness[max_fitness_index])
    x, y = translateDNA(pop)
    print("最优的基因型：", pop[max_fitness_index])
    print("坐标为(x, y):", (x[max_fitness_index], y[max_fitness_index]))
    print("函数最小值为:",10 - F(x[max_fitness_index], y[max_fitness_index]))


if __name__ == "__main__":
    fig = plt.figure(0)
    ax = Axes3D(fig)
    plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
    plot_3d(ax)  # 开始3d绘图

    mean_fitness = []
    best_fitness = []
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))  # matrix (POP_SIZE, DNA_SIZE)
    for _ in range(N_GENERATIONS):  # 迭代N代
        x, y = translateDNA(pop)
        if 'sca' in locals():
            sca.remove()
        sca = ax.scatter(x, y, F(x, y), c='black', marker='o')
        plt.show()
        plt.pause(0.1)
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))
        fitness = get_fitness(pop)
        pop = select(pop, fitness)  # 选择生成新的种群
        mean_fitness.append(np.mean(F(x, y)))
        best_fitness_index = np.argmax(fitness)
        best_fitness.append(F(x[best_fitness_index], y[best_fitness_index]))

    print_info(pop)
    plt.ioff()
    plot_3d(ax)  # 停止3d绘图

    plt.figure(1)
    plt.title("fitness in generations")
    plt.xlabel("iterators", size=14)
    plt.ylabel("fitness", size=14)
    t = np.array([t for t in range(0, N_GENERATIONS)])
    mean_fitness = np.array(mean_fitness)
    best_fitness = np.array(best_fitness)
    plt.plot(t, mean_fitness, label='mean fitness', color='r', linewidth=3)
    plt.plot(t, best_fitness, label='best fitness', color='b', linewidth=3)
    plt.legend(['mean fitness','best fitness'])  # 打出图例
    plt.show()


