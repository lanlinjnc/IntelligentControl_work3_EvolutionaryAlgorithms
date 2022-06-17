#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/14 14:11
# @Author  : lanlin
# reference: http://t.csdn.cn/C9wbn



import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D



class PSO():
    # ----------------------具有惯性权重的PSO参数设置---------------------------------
    def __init__(self, pN, dim, max_iter):  # 初始化类  设置粒子数量  位置信息维度  最大迭代次数
        self.ws = 1.2  # 惯性权重上限
        self.we = 0.9  # 惯性权重下限
        self.phi1 = 2.0
        self.phi2 = 2.0
        self.alpha1 = 0.7
        self.alpha2 = 0.5
        self.pN = pN  # 粒子数量
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置（还要确定取值范围）
        self.Xmax = 1.0
        self.Xmin = -1.0
        self.V = np.zeros((self.pN, self.dim))  # 所有粒子的速度（还要确定取值范围）
        self.Vmax = 1.0
        self.Vmin = -1.0
        self.pbest_place = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置
        self.gbest_place = np.zeros(self.dim)  # 全局最佳位置
        self.pfit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.gfit = 0.0  # 全局最佳适应值，初始化为0

    # ---------------------目标函数-----------------------------
    def F(self, x, y):
        result = x**2 + y**2 - 0.3*np.cos(3*np.pi*x) - 0.4*np.cos(4*np.pi*y) + 0.7
        return result

    def fitness(self, x, y):
        """
            :param x: 染色体上的第一个基因，也就是x
            :param y: 染色体上的第二个基因，也就是y
            :return:  返回适配值。
                      对于求最大值的情况，直接返回函数值；
                      对于求最小值的情况，返回一个较大常数与函数值的差；
        """
        return 10-self.F(x,y)

    # ---------------------初始化种群----------------------------------
    def init_Population(self):
        for i in range(self.pN):  # 遍历所有粒子
            for j in range(self.dim):  # 每一个粒子的纬度
                self.X[i][j] = random.uniform(-1, 1)  # 给每一个粒子的位置赋一个初始随机值（在一定范围内）
                self.V[i][j] = random.uniform(-1, 1)  # 给每一个粒子的速度给一个初始随机值（在一定范围内）

            self.pbest_place[i] = self.X[i]  # 把当前粒子位置作为这个粒子的最优位置
            tmp = self.fitness(self.X[i][0], self.X[i][1])  # 计算这个粒子的适应度值
            self.pfit[i] = tmp  # 当前粒子的适应度值作为个体最优值

            if (tmp > self.gfit):  # 与当前全局最优值做比较并选取更佳的全局最优值
                self.gfit = tmp
                self.gbest_place = self.X[i]

            # ---------------------更新粒子位置----------------------------------

    def iterator(self):

        fig = plt.figure()
        ax = Axes3D(fig)
        plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行
        self.plot_3d(ax)

        mean_fitness = []
        best_fitness = []

        for t in range(self.max_iter):
            # w = self.ws - (self.ws - self.we) * (t / self.max_iter)  # 惯性权重
            w = 1  # 取消惯性权重
            for i in range(self.pN):

                # 更新速度
                part1 = self.phi1 * random.uniform(0, self.alpha1) * (self.pbest_place[i] - self.X[i])
                part2 = self.phi2 * random.uniform(0, self.alpha2) * (self.gbest_place - self.X[i])
                self.V[i] = w * self.V[i] + part1 + part2
                # 限定速度范围
                if self.V[i][0] > self.Vmax:
                    self.V[i][0] = self.Vmax
                elif self.V[i][0] < self.Vmin:
                    self.V[i][0] = self.Vmin
                if self.V[i][1] > self.Vmax:
                    self.V[i][1] = self.Vmax
                elif self.V[i][1] < self.Vmin:
                    self.V[i][1] = self.Vmin

                # 更新位置
                self.X[i] = self.X[i] + self.V[i]
                # 限定位置范围
                if self.X[i][0] > self.Xmax:
                    self.X[i][0] = self.Xmax
                elif self.X[i][0] < self.Xmin:
                    self.X[i][0] = self.Xmin
                if self.X[i][1] > self.Xmax:
                    self.X[i][1] = self.Xmax
                elif self.X[i][1] < self.Xmin:
                    self.X[i][1] = self.Xmin

                # 更新gbest\pbest
                tempfit = self.fitness(self.X[i][0], self.X[i][1])
                if (tempfit > self.pfit[i]):  # 更新个体最优
                    self.pbest_place[i] = self.X[i]
                    self.pfit[i] = tempfit

                if (tempfit > self.gfit):  # 更新全局最优
                    self.gbest_place = self.X[i]
                    self.gfit = tempfit

            mean_fitness.append(np.mean(self.fitness(self.X[0], self.X[1])))
            best_fitness.append(self.gfit)
            zb = self.gfit
            print('历史最优值为：', 10 - zb)  # 输出最优值
            xb = self.gbest_place[0]
            yb = self.gbest_place[1]
            print('历史最优位置为：', xb, yb)

            if 'sca' in locals():
                sca.remove()
            sca = ax.scatter(self.X[0], self.X[1], 0.1+    # 加上0.1仅是为了显示需要
                  self.fitness(self.X[0], self.X[1]),  c='black', marker='o')

            plt.show()
            plt.pause(0.1)  # 秒
            # plt.cla()  # 清除画板上的图，但不清除画板

        plt.ioff()
        self.plot_3d(ax)  # 停止3d绘图

        return best_fitness, mean_fitness, 10-zb, xb, yb

    def plot_3d(self, ax):
        X = np.linspace(-1.0, 1.0, 100)
        Y = np.linspace(-1.0, 1.0, 100)
        X, Y = np.meshgrid(X, Y)
        Z = self.fitness(X, Y)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.pause(1)
        plt.show()


if __name__=="__main__":
    # ----------------------程序执行-----------------------
    pop_num = 100
    data_dim = 2
    generation = 200
    my_pso = PSO(pN=pop_num, dim=data_dim, max_iter=generation)
    my_pso.init_Population()
    best_fitness, mean_fitness, z1, x1, y1 = my_pso.iterator()

    plt.figure(1)
    plt.title("fitness in generations")
    plt.xlabel("iterators", size=14)
    plt.ylabel("fitness", size=14)
    t = np.array([t for t in range(0, generation)])
    mean_fitness = np.array(mean_fitness)
    best_fitness = np.array(best_fitness)
    plt.plot(t, mean_fitness, label='mean fitness', color='r', linewidth=3)
    plt.plot(t, best_fitness, label='best fitness', color='b', linewidth=3)
    plt.legend(['mean fitness', 'best fitness'])  # 打出图例
    plt.show()
