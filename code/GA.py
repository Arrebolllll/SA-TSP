import time
import pandas as pd
import numpy as np
import random
import heapq
import math
import matplotlib.pyplot as plt


class Path:
    def __init__(self, path):
        self.path = path
        self.cost = get_distance(path)

    def __lt__(self, other):
        if self.cost < other.cost:
            return True
        else:
            return False


def ramdom_init(ramdom_size):
    """
    随机初始化种群中的个体
    """
    ramdom_population = []
    while len(ramdom_population) < ramdom_size:  # 不用for主要是怕重复
        path = [i for i in range(city_num)]
        random.shuffle(path)  # 随机打乱
        ele = Path(path)
        ramdom_population.append(ele)
    return ramdom_population


def greedy_init(greedy_size):
    """
    贪婪算法初始化一些个体
    """
    greedy_population = []
    while len(greedy_population) < greedy_size:
        greedy_path = []
        current_city = random.randint(0, city_num - 1)  # 随便选一个城市作为出发点，贪婪算法寻求以这点出发的最短路径
        greedy_path.append(current_city)
        remain_city = [i for i in range(city_num)]
        remain_city.remove(current_city)
        while len(remain_city) > 0:
            min = float('inf')
            next_city = None
            for City in remain_city:
                temp_distance = distance[current_city][City]
                if temp_distance < min:
                    min = temp_distance
                    next_city = City
            greedy_path.append(next_city)
            current_city = next_city
            remain_city.remove(current_city)
        greedy_population.append(Path(greedy_path))
    return greedy_population


def init():
    greedy_size = 2  # 通过贪婪算法生成的个体数
    ramdom_size = 98  # 随机生成的个体数
    _population = greedy_init(greedy_size) + ramdom_init(ramdom_size)
    _population.sort()
    return _population, greedy_size + ramdom_size


def get_distance(currentPath):
    # 评判，distance越小越好
    dis = 0
    for index in range(city_num - 1):
        # print(currentPath[index], currentPath[index + 1])
        dis += distance[currentPath[index]][currentPath[index + 1]]

    dis += distance[currentPath[0]][currentPath[-1]]
    return dis


def select():
    parents = []
    for _ in range(3):  # 选三个parents
        temp_parents = []
        for i in range(10):  # 随机抽十个
            idx = random.randint(0, population_size - 1)
            temp_parents.append(population[idx])
        parents.append(min(temp_parents))  # 选十个中最小的那个
    return parents


def cross(Path1, Path2):
    # 切片复制速度更快
    parent1 = Path1.path[:]
    parent2 = Path2.path[:]
    # 随便选两个位置
    posA = random.randint(0, city_num - 1)
    posB = random.randint(0, city_num - 1)
    # 两个位置自然是不能相等的
    while posA == posB:
        posB = random.randint(0, city_num - 1)
    # 保持posA在posB左边
    if posA > posB:
        posB, posA = posA, posB
    select_part1 = parent1[posA:posB + 1]
    select_part2 = parent2[posA:posB + 1]
    # 根据概率选择交叉策略
    coin = random.random()
    if coin <= 0.5:
        # PMX
        child = parent2[:]
        child[posA: posB + 1] = select_part1
        for index in range(len(child)):
            if posA <= index <= posB:  # 在被交换区域外的城市寻找映射关系，在交换区内的直接跳过
                continue
            if child[index] in select_part1:
                temp = select_part2[select_part1.index(child[index])]
                while temp in select_part1:
                    temp = select_part2[select_part1.index(temp)]
                child[index] = temp  # 根据映射关系修改冲突的城市
    # elif coin <= 0.55:
    #     # OX
    #     child = [-1] * len(parent1)
    #     child[posA:posB + 1] = select_part1
    #     for index1 in range(len(child)):
    #         if posA <= index1 <= posB:
    #             continue
    #         for index2 in range(len(parent2)):
    #             if parent2[index2] not in child:
    #                 child[index1] = parent2[index2]
    #                 break
    # elif coin <= 0.6:
    #     # PBX
    #     child = [-1] * len(parent1)
    #     index = 0
    #     while index < int(len(parent1) / 2):
    #         chosen_index = random.randint(0, len(parent1) - 1)
    #         if child[chosen_index] != -1:
    #             continue
    #         child[chosen_index] = parent1[chosen_index]
    #         index += 1
    #     for index1 in range(len(child)):
    #         if child[index1] != -1:
    #             continue
    #         for index2 in range(len(parent2)):
    #             if parent2[index2] not in child:
    #                 child[index1] = parent2[index2]
    #                 break
    else:
        # CX
        # find the cycle between parents
        cycle = []
        start = parent1[0]
        end = parent2[0]
        cycle.append(start)
        while end != start:
            cycle.append(end)
            end = parent2[parent1.index(end)]
        # produce proto-child by copying gene values in cycle from parent 1
        child = parent1[:]
        cycle_point = cycle[:]
        if len(cycle_point) < 2:
            cycle_point = random.sample(parent1, 2)
        index3 = 0
        for index1 in range(len(parent1)):
            if child[index1] in cycle_point:
                continue
            else:
                for index2 in range(index3, len(parent2)):
                    if parent2[index2] in cycle_point:
                        continue
                    else:
                        child[index1] = parent2[index2]
                        index3 = index2 + 1
                        break
    return Path(child)


def mutation(children):
    mutation_rate = 0.5
    for idx in range(len(children)):
        if random.random() < mutation_rate:  # 要变了
            child = children[idx]  # 取出要变的孩子
            newPath = child.path[:]  # 孩子对应的路径
            # 随便找路径中的两个位置
            posA = random.randint(0, city_num - 2)
            posB = random.randint(posA + 1, city_num - 1)
            # 依据概率选择对应的变异操作
            coin = random.random()
            if coin < 0.25:
                # 交换两个位置的元素
                newPath[posA], newPath[posB] = newPath[posB], newPath[posA]
            if 0.25 <= coin < 0.5:
                # 打乱两个位置之间的元素
                part = newPath[posA: posB + 1]
                random.shuffle(part)
                newPath[posA:posB + 1] = part
            if 0.5 <= coin < 0.75:
                # 逆转两个位置之间的元素
                part = newPath[posA:posB + 1]
                part.reverse()
                newPath[posA:posB + 1] = part
            if coin >= 0.75:
                # 把后面位置的元素放到前面位置的后面
                newPath[posA + 1], newPath[posB] = newPath[posB], newPath[posA + 1]
            newchild = Path(newPath)  # 变异后生成的新娃
            children[idx] = newchild  # 更新
    return children


def GA():
    global population  # 种群
    t = 0  # 已进化的代数
    evolution_times = 4000  # 代数
    the_one = population[0]  # 最短的那个个体
    DisRecord = []  # 记录路径变化
    while t < evolution_times:  # 收敛条件
        last_best = population[0]  # 精英策略，如果上一代最好的比这一代最差的好，就保留上一代最好的
        parents = select()  # 选择操作找出父母
        needed_num = population_size - len(parents)  # 选择操作找出的父母个数不是一定的，需要的孩子数目是种群数目-父母数目
        children = []  # 孩子列表
        while len(children) < needed_num:  # 通过交叉操作得到孩子
            Path1, Path2 = random.sample(parents, 2)
            child1 = cross(Path1, Path2)
            children.append(child1)
            child2 = cross(Path2, Path1)
            children.append(child2)
        children = mutation(children)  # 变异操作
        population = parents + children  # 新种群
        population.sort()  # 对新种群排序
        if population[-1].cost > last_best.cost:  # 精英策略
            population[-1] = last_best
            population.sort()

        if population[0].cost < the_one.cost:  # 更新最优值
            the_one = population[0]

        DisRecord.append(the_one.cost)  # 记录最优个体距离变换
        t += 1
        print(t, "generation: ", the_one.cost)
        # 查看种群每个个体情况
        # for i in range(population_size):
        #     print("%.1f" % population[i].cost, end=' ')
        # print()
    return the_one, DisRecord, t


def drawpath(the_one):
    bestpath = the_one.path
    dist = the_one.cost
    x = [0 for col in range(city_num + 1)]
    y = [0 for col in range(city_num + 1)]
    for i in range(0, city_num):
        x[i] = city_x[bestpath[i]]
        y[i] = city_y[bestpath[i]]
    x[city_num] = x[0]
    y[city_num] = y[0]

    # plt.xlim(0, 800)
    # plt.ylim(0, 800)
    plt.plot(x, y, marker='o', mec='g', mfc='w', label='path')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    title = "Total distance:" + str(dist)
    plt.title(title)
    plt.show()


def drawChange(record, title):
    x = [0 for col in range(times)]
    y = [0 for col in range(len(record))]
    for i in range(0, times):
        x[i] = i
    for j in range(0, len(record)):
        y[j] = record[j]
    plt.xlim()
    plt.ylim()
    plt.plot(x, y, marker='', mec='r', mfc='w', label='path')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # 从文档中载入数据
    df = pd.read_csv('F:/AI/Lab/report/lab5/ch150.tsp', sep=" ", skiprows=6, header=None)
    city = np.array(df[0][0:len(df) - 1])  # 最后一行为EOF，不读入
    city_list = city.tolist()
    print(city_list)
    city_x = np.array(df[1][0:len(df) - 1])
    city_y = np.array(df[2][0:len(df) - 1])
    city_location = list(zip(city_x, city_y))
    # print(city_location)
    city_num = len(city_list)
    # print(city_num)

    # 初始化distance矩阵
    distance = [[0 for col in range(city_num)] for row in range(city_num)]
    for i in range(city_num):
        for j in range(i, city_num):
            dx = pow(city_x[i] - city_x[j], 2)
            dy = pow(city_y[i] - city_y[j], 2)
            distance[i][j] = distance[j][i] = pow(dx + dy, 0.5)
    distance = tuple(distance)

    # 初始化种群，全部都是随机生成的，由cost小到大，方便选择
    population, population_size = init()

    time1 = time.time()
    best, Disrecord, times = GA()
    time2 = time.time()
    print(time2 - time1)

    drawpath(best)
    drawChange(Disrecord, 'Distance')
