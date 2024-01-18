import time
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt


def get_distance(currentPath):
    dis = 0
    for index in range(0, city_num - 1):
        dis += distance[currentPath[index]][currentPath[index + 1]]
    dis += distance[currentPath[city_num - 1]][currentPath[0]]
    return dis


def get_new_path(currentPath):
    """
    path：当前的path
    要生成新的path
    可以交换两个位置
    可以反转两个位置之间的元素
    可以将这个位置的元素插入到后一个位置的元素后
    """
    coin = random.random()

    newPath = currentPath[:]
    if coin < 0.5:
        # 随机生成待翻转的两个位置
        posA = random.randint(0, city_num - 1)
        posB = random.randint(0, city_num - 1)
        # 两个位置自然是不能相等的
        while posA == posB:
            posB = random.randint(0, city_num - 1)
        to_be_reverse = newPath[posA: posB + 1]
        to_be_reverse.reverse()
        newPath[posA: posB + 1] = to_be_reverse

    elif coin < 0.8:
        # 随机生成待交换的两个位置,闭区间
        posA = random.randint(0, city_num - 1)
        posB = random.randint(0, city_num - 1)
        # 两个位置自然是不能相等的
        while posA == posB:
            posB = random.randint(0, city_num - 1)
        # 交换
        newPath[posA], newPath[posB] = newPath[posB], newPath[posA]
    else:
        posA = random.randint(0, city_num - 1)
        posB = random.randint(0, city_num - 1)
        # 两个位置自然是不能相等的
        while posA == posB:
            posB = random.randint(0, city_num - 1)
        if posB + 1 < city_num:
            newPath[posB + 1], newPath[posA] = newPath[posA], newPath[posB + 1]
        else:
            newPath[posA + 1], newPath[posB] = newPath[posB], newPath[posA + 1]
    return newPath


def SA():
    """
    退火函数
    """
    T0 = 5000  # 初温
    T_end = 1e-6  # 终温
    decline_rate = 0.998  # 降温率
    up_rate = 0.5  # 升温率
    inner_loops = 3000  # 内循环次数
    same = 0
    global times  # 时间点记录
    global path, shortestDis, thePath
    current_distance = shortestDis
    pathdic = set()  # 保存已经有过的路径，末端加速
    T = T0
    while T > T_end:  # 外循环
        for _ in range(inner_loops):  # 内循环
            new_path = get_new_path(path)
            if hash(tuple(new_path)) in pathdic:  # 对于重复的状态不需要重复生成
                continue
            pathdic.add(hash(tuple(new_path)))
            new_distance = get_distance(new_path)
            dDistance = new_distance - current_distance  # 计算新的路径是否比现在的路径短
            if dDistance < 0:  # 新的比现在的短
                # 接受新的
                if current_distance == new_distance:
                    same += 1
                current_distance, path = new_distance, new_path
                if new_distance < shortestDis:
                    shortestDis, thePath = new_distance, new_path[:]
            else:  # 新的比现在的长，以一定的概率去接受
                p = math.exp(-dDistance / T)
                if random.random() < p:
                    if current_distance == new_distance:
                        same += 1
                    path, current_distance = new_path, new_distance

        times += 1
        if same >= 20:  # 路径太久不变，重复升温，增加扰动
            T = T * (1 + up_rate)
            same = 0
        else:  # 完成一次外循环，降温
            T = T * decline_rate
        DisRecord.append(current_distance)  # 记录路径变化
        TempRecord.append(T)  # 记录温度变化
        if times % 10 == 0:
            print(T, shortestDis)


def drawPath(bestpath, dist):
    x = [0 for col in range(city_num + 1)]
    y = [0 for col in range(city_num + 1)]
    for i in range(city_num):
        x[i] = city_x[bestpath[i]]
        y[i] = city_y[bestpath[i]]
    x[city_num] = x[0]
    y[city_num] = y[0]

    plt.plot(x, y, marker='o', mec='y', mfc='w', label='path')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    title = "SA Total distance:" + str(dist)
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
    # print(city_list)
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
    # 给出一个随机解
    path = list(range(city_num))
    random.shuffle(path)  # 随机打乱

    # 全局最优解
    thePath = path
    # 全局最优距离
    shortestDis = get_distance(path)
    # print(shortestPath)

    # 记录路长的变换
    DisRecord = []
    TempRecord = []
    t1 = time.time()
    times = 0
    SA()
    t2 = time.time()
    print(t2 - t1)
    drawPath(thePath, shortestDis)
    drawChange(DisRecord, 'Distance')
    drawChange(TempRecord, 'Temperature')
