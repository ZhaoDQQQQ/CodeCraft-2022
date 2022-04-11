# -*- coding: utf-8 -*-
import csv
import copy
import configparser
import time
import numpy as np


class Point:
    # 客户节点 类
    def __init__(self, name):
        # name标识符
        self.name = name
        # 各时刻带宽需求list
        self.demand = []
        # 可连接的边缘节点字典
        self.qos_server = {}
        # 各时刻分配方案
        self.distribution_plan = []
        # 优先级，可连接边缘节点越多  数值越大
        self.priority = 0

    def add_demand(self, demand):
        # 添加一个时刻的带宽需求
        self.demand.append(demand)

    def add_qos_server(self, name, server):
        # 添加可连接的边缘节点
        self.qos_server[name] = server
        self.priority += 1

    def get_renew_server_list(self, num):
        # 返回在当前条件下  该客户节点可以影响到的边缘节点
        renew_server_list = []
        for name in self.qos_server.keys():
            if name not in server_points_ranking:
                continue
            index = server_points_ranking.index(name)
            if server_matrix[num][index] != 0:
                renew_server_list.append(index)
        return renew_server_list

    def get_needed(self, name, max_bandwidth, num):
        # 进行分配   贪！能贪多少是多少！并记录下来
        needed = self.demand[num]
        if needed == 0:
            return max_bandwidth
        if needed <= max_bandwidth:
            max_bandwidth -= needed
            temp = "<" + name + "," + str(needed) + ">"
            self.distribution_plan[num].append(temp)
            needed = 0
            index = points_to_done[num].index(self.name)
            del points_to_done[num][index]
        else:
            needed -= max_bandwidth
            temp = "<" + name + "," + str(max_bandwidth) + ">"
            self.distribution_plan[num].append(temp)
            max_bandwidth = 0
        self.demand[num] = needed
        # print(temp)
        return max_bandwidth

    def noisy_get_needed(self, name, max_bandwidth, num):
        # 进行分配   贪！能贪多少是多少！并记录下来
        needed = self.demand[num]
        if needed == 0:
            return max_bandwidth, needed
        if needed <= max_bandwidth:
            temp = needed
            max_bandwidth -= needed
            needed = 0
            self.demand[num] = needed
            index = points_to_done[num].index(self.name)
            del points_to_done[num][index]
            return max_bandwidth, temp
        else:
            needed -= max_bandwidth
            self.demand[num] = needed
            return 0, max_bandwidth

    def average_get_needed(self, name, max_bandwidth, num):
        # 客户节点进行平均分配
        # a客户节点尚可分配的边缘节点具有3个   则分配1/3给A边缘节点
        needed = self.demand[num]
        if needed == 0:
            return max_bandwidth, needed
        # 平均分成几份呢？
        divisor = 1
        test_server_names = list(self.qos_server.keys())
        test_server_names.remove(name)
        for test_server_name in test_server_names:
            test_point = self.qos_server[test_server_name]
            if num in test_point.demand_sequence.keys():
                pass
            else:
                divisor += 1
        average_needed = int(needed/divisor)
        # 进行需求分配
        if average_needed <= max_bandwidth:
            max_bandwidth -= average_needed
            needed = needed - average_needed
            self.demand[num] = needed
            if needed == 0:
                index = points_to_done[num].index(self.name)
                del points_to_done[num][index]
            return max_bandwidth, average_needed
        else:
            needed -= max_bandwidth
            self.demand[num] = needed
            return 0, max_bandwidth
        # print(temp)


class ServerPoint:
    def __init__(self, name, bandwidth):
        # name标识符
        self.name = name
        # 带宽上限
        self.bandwidth = bandwidth
        # 带宽序列  好吧  最后也没有用到   键是时刻   值是分配带宽量
        self.demand_sequence = {}
        # 可连接的客户节点字典
        self.qos_point = {}
        # 优先级，可连接客户节点越多  数值越大
        self.priority = 0
        # 已连接的客户节点的在分配带宽时的顺序，靠前的优先被分配
        self.order_list = []
        # 5%的时刻的实际数值  比如 1000*0.05-1 = 49
        self.flag = 0
        # 静态flag
        self.static_flag = 0

        # 该回合被执行的客户节点，便于更新对应的矩阵值
        self.done_points = []
        # 5%里面最小的数值
        self.max_4th = 2.0
        # 5%里面的所有数值
        self.max_list = []
        # 95%里面最大的数值
        self.max_5th = 1000000
        # 95%里面最大的数值对应的时刻
        self.max_5th_index = 0
        # 95%里面每一时刻对应的带宽分配
        self.dic95 = {}

    def add_qos_point(self, name, point):
        # 添加可连接的客户节点
        self.qos_point[name] = point
        self.priority += 1

    def get_left(self):
        # 获取该边缘节点每一时刻的最大带宽需求
        left = [0] * time_length
        left = np.array(left)
        for name in self.order_list:
            used_x = np.array(self.qos_point[name].demand)
            left = left + used_x
        return left

    def reget_left(self, num):
        # 获取该边缘节点某一时刻的最大带宽需求
        left = 0
        for name in self.order_list:
            left += self.qos_point[name].demand[num]
        return left

    def get_used(self, num):
        used = 0
        for name in self.order_list:
            used += self.qos_point[name].demand[num]
        return used/(self.bandwidth + 0.1)

    def get_ranking(self, temp_list):
        # 获取该边缘节点的连接的客户节点与temp_list中数据的重合数量
        ranking = 0
        for point_name in self.qos_point.keys():
            if point_name in temp_list:
                ranking += 1
                temp_list.remove(point_name)
        return ranking, temp_list

    def renew_server_matrix(self, num):
        # 更新server_matrix矩阵
        global server_matrix
        # flag等于0  代表没有位置了   该边缘节点不用考虑了  直接删掉吧
        if self.flag == 0:
            index = server_points_ranking.index(self.name)
            server_matrix = np.delete(server_matrix, index, axis=1)
            server_points_ranking.remove(self.name)
        else:
            index = server_points_ranking.index(self.name)
            server_matrix[num][index] = 0

        # 记录下来被影响到的边缘节点
        renew_server_list = []
        for name in self.order_list:
            if name not in self.done_points:
                continue
            renew_server_list += self.qos_point[name].get_renew_server_list(num)
        # print(len(renew_server_list))
        # 去除重复节点
        renew_server_list = list(set(renew_server_list))
        # print(len(renew_server_list))
        # 更新矩阵中被影响到的位置的数值
        for index in renew_server_list:
            renew_server_name = server_points_ranking[index]
            renew_server_point = server_points[renew_server_name]
            left = renew_server_point.reget_left(num)
            server_matrix[num][index] = left

    def renew_max_5th(self):
        # 获取95%位置的带宽值以及其时刻数值
        for name_key in self.dic95.keys():
            if self.dic95[name_key] < self.max_5th:
                self.max_5th = self.dic95[name_key]
                self.max_5th_index = name_key

    def run_one(self, num):
        # 进行分配   完全分配  能够贪到多少就多少
        to_do_points = points_to_done[num]
        self.done_points = []
        self.flag -= 1
        left = self.bandwidth
        # 开始贪！
        for name in self.order_list:
            if name not in to_do_points:
                continue
            left = self.qos_point[name].get_needed(self.name, left, num)
            self.done_points.append(name)
            if left == 0:
                break
        # 记录一下贪的结果
        self.demand_sequence[num] = left
        uesd_percentage = (self.bandwidth - left) / (self.bandwidth + 0.1)
        self.max_list.append(uesd_percentage)
        if self.max_4th == 2.0:
            self.max_4th = uesd_percentage
        elif uesd_percentage < self.max_4th:
            self.max_4th = uesd_percentage
        # 更新server_matrix矩阵  因为部分客户节点的需求被分配了  边缘节点的相应接收需求也会同步变化
        self.renew_server_matrix(num)

    def noisy_run_one(self, num, index):
        # 对噪声边缘节点  进行需求分配
        # 分配方案  能分配多少分配多少
        to_do_points = points_to_done[num]
        self.done_points = []
        left = self.bandwidth
        # 开始贪！
        for name in self.order_list:
            if name not in to_do_points:
                continue
            left, used = self.qos_point[name].noisy_get_needed(self.name, left, num)
            if used == 0:
                pass
            else:
                gradient_descent_matrix[num][index].append([used, name])
            if left == 0:
                break
        # 记录一下贪的结果
        self.demand_sequence[num] = left
        self.dic95[num] = left
        # uesd_percentage = (self.bandwidth - left) / (self.bandwidth + 0.1)
        # 记录下该95%中最大的数值
        if self.max_5th == 1000000:
            self.max_5th = left
            self.max_5th_index = num
        elif left < self.max_5th:
            self.max_5th = left
            self.max_5th_index = num


    def average_run_one(self, num, index):
        # 对该边缘节点进行需求分配   采用平均分配的思想
        # A边缘节点连接a客户节点，a客户节点尚可分配的边缘节点具有3个   则分配1/3给A边缘节点
        to_do_points = points_to_done[num]
        self.done_points = []
        left = self.bandwidth
        # 开始平均分配
        for name in self.order_list:
            if name not in to_do_points:
                continue
            left, used = self.qos_point[name].average_get_needed(self.name, left, num)
            if used == 0:
                pass
            else:
                gradient_descent_matrix[num][index].append([used, name])
            if left == 0:
                break
        # 记录相关信息
        self.demand_sequence[num] = left
        self.dic95[num] = left
        # uesd_percentage = (self.bandwidth - left) / (self.bandwidth + 0.1)
        if self.max_5th == 1000000:
            self.max_5th = left
            self.max_5th_index = num
        elif left < self.max_5th:
            self.max_5th = left
            self.max_5th_index = num


def set_server_flag():
    # 根据总时间序列长度计算可用（可舍弃）的5%的值
    flag = int(time_length*0.05) - 1
    # print(flag)
    for point in server_points.values():
        point.flag = flag
        point.static_flag = flag

    # 顺便把客户节点的分配方案列表也建立好   方便后期查询
    for point in points.values():
        for i in range(time_length):
            point.distribution_plan.append([])
        # print(point.distribution_plan)


def order_points():
    # 对边缘节点可以连接到的客户节点进行排序
    # 客户节点所能连接到的边缘节点越少，越靠前
    # 为了避免 a仅与A相连，但是A边缘节点分配满了，a无处可连接的情况
    for server_point in server_points.values():
        order_list = []
        for point in server_point.qos_point.values():
            order_list.append([point.priority, point.name])
        order_list.sort()
        server_point.order_list = [j for (i, j) in order_list]


def new_server_matrix(server_mame_list):
    # 构建server_matrix初始矩阵
    # server_matrix 时刻数*边缘节点数 的矩阵   每个元素代表对应时刻，对应边缘节点所能接收到的最大需求
    global server_matrix
    server_points_ranking = []
    for i in range(len(server_mame_list)):
        name = server_mame_list[i]
        server_point = server_points[name]
        # 如果该边缘节点不与任何客户节点相连（qos约束下），则无需考虑该节点
        if server_point.priority == 0:
            continue
        server_points_ranking.append(name)
        # 获取该边缘节点所有时刻的需求
        left = server_point.get_left()
        # 拼接到server_matrix矩阵上
        server_matrix = np.column_stack((server_matrix, left))
    return server_points_ranking


def get_noisy_list():
    # 获取进行噪声聚集的边缘节点
    # 1.保证每个客户节点都有边缘节点进行连接
    # 2.保证最后的边缘节点的数目尽可能少
    temp_list = copy.deepcopy(name_list)
    noisy_list = []
    while temp_list != []:
        order_list = []
        for i in range(len(server_points_ranking)):
            name = server_points_ranking[i]
            if name in noisy_list:
                continue
            server_point = server_points[name]
            # 获取该边缘节点的连接的客户节点与temp_list中数据的重合数量以及删去重合后的temp_list
            ranking, new_temp_list = server_point.get_ranking(copy.deepcopy(temp_list))
            order_list.append([ranking, name, new_temp_list])
        order_list.sort(reverse=True)
        noisy_list.append(order_list[0][1])
        temp_list = copy.deepcopy(order_list[0][2])
        # print(temp_list)
    return noisy_list


#############################################################################################################
#############################################################################################################
#############################################################################################################
def read_demand():
    # read point data   读取客户节点流量需求
    with open(path0, 'r') as f:
        reader = csv.reader(f)
        reader = list(reader)
        row = reader[0]
        for p in row[1:]:
            points[p] = Point(p)
            name_list.append(p)

        for row in reader[1:]:
            mtime.append(row[0])
            for j in range(len(name_list)):
                p = name_list[j]
                d = row[j + 1]
                points[p].add_demand(int(d))


def read_site_bandwidth():
    # read server point data     读取边缘节点带宽需求
    with open(path1, 'r') as f:
        reader = csv.reader(f)
        reader = list(reader)
        for row in reader[1:]:
            server_points[row[0]] = ServerPoint(row[0], int(row[1]))


def read_ini():
    # read qos max       读取qos约束
    cf = configparser.ConfigParser()
    cf.read(path2)
    qos_max = int(cf.get("config", "qos_constraint"))
    # print(qos_max)
    del cf
    return qos_max


def read_qos_matrix(qos_max):
    # read qos matrix   读取客户节点与边缘节点的qos约束关系
    with open(path3, 'r') as f:
        reader = csv.reader(f)
        reader = list(reader)
        row = reader[0]
        name_list = []
        for p in row[1:]:
            name_list.append(p)

        for row in reader[1:]:
            server_name = row[0]
            for j in range(len(name_list)):
                qos = int(row[j + 1])
                point_name = name_list[j]
                # 只有满足qos约束的才会被添加到对应客户节点/边缘节点的属性中
                if qos < qos_max:
                    server_points[server_name].add_qos_point(point_name, points[point_name])
                    points[point_name].add_qos_server(server_name, server_points[server_name])
#############################################################################################################
#############################################################################################################
#############################################################################################################


if __name__ == '__main__':
    system = 0
    if system == 0:
        # windows下的路径
        path0 = '.\\data\\demand.csv'
        path1 = '.\\data\\site_bandwidth.csv'
        path2 = '.\\data\\config.ini'
        path3 = '.\\data\\qos.csv'
        path4 = ".\\output\\solution.txt"
    else:
        #  ubuntu系统下的路径
        path0 = '/data/demand.csv'
        path1 = '/data/site_bandwidth.csv'
        path2 = '/data/config.ini'
        path3 = '/data/qos.csv'
        path4 = "/output/solution.txt"

    t = time.time()
    # 保存所有时刻   不过最后也没有用到
    mtime = []
    time_length = 0
    # 字典，存储所有的point  键为客户节点名字   值为point类
    points = {}
    # 记录所有point名字  便于按照固定次序查询
    name_list = []
    # 字典，存储所有的serverPoint  键为边缘节点名字   值为ServerPoint类
    server_points = {}
    # 约束条件
    qos_max = 0

    # 读取客户节点的时间序列上的需求   demand.csv
    read_demand()
    time_length = len(mtime)

    # 读取边缘节点的带宽约束   site_bandwidth.csv
    read_site_bandwidth()

    # 读取qos约束条件   config.ini
    qos_max = read_ini()

    # 读取客户节点与边缘节点的qos约束关系    qos.csv
    read_qos_matrix(qos_max)

    # 预处理阶段
    # 设定相关flag，具体信息，跳转看函数介绍
    set_server_flag()
    # 对边缘节点所能连接到的客户节点进行排序，具体信息，跳转看函数介绍
    order_points()

    # 客户节点所有节点名字
    points_ranking = list(points.keys())
    # 边缘节点所有节点名字
    server_name_list = list(server_points.keys())
    # print(points_ranking)
    # print(server_points_ranking)

    # server_matrix 大小为 时刻数*边缘节点数 的矩阵   每个元素代表对应时刻，对应边缘节点所能接收到的最大需求
    server_matrix = np.zeros((time_length, 1))
    # 更新矩阵，填充每个元素的值   返回值时  矩阵每一列对应的边缘节点名字构成的列表
    # 若边缘节点不与客户节点连接，则不被包含进矩阵(以及列表)，所以不等同于server_name_list
    server_points_ranking = new_server_matrix(server_name_list)
    server_matrix = np.delete(server_matrix, 0, axis=1)

    # 待调整矩阵   大小为 时刻数*边缘节点数 的矩阵 每个元素代表对应时刻，对应边缘节点被分配的需求
    # 这里95%-100%百分位带宽为None，不进行调整
    # 主要对95百分位带宽进行调整，让其尽可能的小，从而降低成本
    gradient_descent_matrix = []
    temp = []
    for j in range(len(server_points_ranking)):
        temp.append([])
    for i in range(time_length):
        gradient_descent_matrix.append(copy.deepcopy(temp))
    del temp

    server_points_ranking_temp = copy.deepcopy(server_points_ranking)

    # 每个时刻待处理的客户节点   一个查询列表  降低时间复杂度用
    points_to_done = []
    for i in range(time_length):
        points_to_done.append(points_ranking[:])

    # 第一步：对每个边缘节点选择出5%可以舍去的时刻，并进行分配
    while True:
        # 求出每个边缘节点的所有时刻的带宽需求之和
        sum_matrix = np.sum(server_matrix, axis=0)
        # 求出最大总带宽对应边缘节点
        server_index = np.argmax(sum_matrix)
        server_name = server_points_ranking[server_index]
        server_point = server_points[server_name]
        # 求出该边缘节点对应的的最大的5%时刻
        # 循环，直至用完5%，或者无客户节点可分配
        while server_point.flag > 0:
            col_server_array = server_matrix[:, server_index]
            # 选择出带宽需求最大的时刻
            num = np.argmax(col_server_array)
            max_x = col_server_array[num]
            if max_x == 0:
                break
            # 进行带宽分配，并且更新矩阵，该时刻的其他边缘节点的带宽需求会有变化
            server_point.run_one(num)

        # 判断是否还能进行分配
        if server_matrix.size == 0:
            break
        max_x = server_matrix.max()
        if max_x == 0:
            break

    # 重置一下
    server_points_ranking = copy.deepcopy(server_points_ranking_temp)

    # 获取负责聚集噪声的边缘节点   该部分节点采用全部贪婪的方法  能分配多少分配多少
    # 其他节点采用平均分配的方法
    # 经实践检验  效果更好
    # 理论解释的话  可以理解为尽可能将峰值集中到了少数几个点上，根据大数定理，这几个点的数据始终可以保持较大数值  减小了波动
    noisy_list = get_noisy_list()

    server_points_ranking = copy.deepcopy(server_points_ranking_temp)
    # noisy_list = noisy_list[:int(len(noisy_list) * 0.75)]

    # 对噪声边缘节点  进行需求分配
    # 分配方案  能分配多少分配多少
    for num in range(time_length):
        for index in range(len(noisy_list)):
            server_name = noisy_list[index]
            real_server_index = server_points_ranking.index(server_name)
            server_point = server_points[server_name]
            if server_point.priority == 0:
                continue
            if num in server_point.demand_sequence.keys():
                gradient_descent_matrix[num][real_server_index] = None
                continue
            server_point.noisy_run_one(num, real_server_index)
    pass_noisy_list = copy.deepcopy(server_points_ranking_temp)
    for noisy_name in noisy_list:
        pass_noisy_list.remove(noisy_name)
    # 对剩余边缘节点进行需求分配   采用平均分配的思想
    # a客户节点连接3个可分配的边缘节点  则各分配1/3
    for num in range(time_length):
        for index in range(len(pass_noisy_list)):
            server_name = pass_noisy_list[index]
            real_server_index = server_points_ranking.index(server_name)
            server_point = server_points[server_name]
            if server_point.priority == 0:
                continue
            if num in server_point.demand_sequence.keys():
                gradient_descent_matrix[num][real_server_index] = None
                continue
            server_point.average_run_one(num, real_server_index)

    # 梯度下降  一点点调整  降低边缘节点95%位置的带宽大小
    gd_done_list = []
    server_points_ranking_list = []
    for server_name in server_points_ranking:
        server_point = server_points[server_name]
        server_points_ranking_list.append([server_point.max_5th, server_name])
    # 排序边缘节点  95%位置带宽越小  位置越靠前
    server_points_ranking_list.sort(reverse=True)
    pass_point_dic = {}
    # 控制时间  时间总限制300s
    while time.time() - t <= 293:
        if server_points_ranking_list == []:
            break
        # print(server_points_ranking)
        # 优先选择95%位置带宽最小的边缘节点进行梯度下降
        server_name_from = server_points_ranking_list[0][1]
        del server_points_ranking_list[0]
        server_point_from = server_points[server_name_from]
        num = server_point_from.max_5th_index
        max_5th = server_point_from.max_5th

        if max_5th == 1000000:
            gd_done_list.append(server_name_from)
            continue
        if server_name_from in gd_done_list:
            continue
        # total_loss记录总的下降数值
        total_loss = 0
        temp_row = gradient_descent_matrix[num]
        real_server_index = server_points_ranking.index(server_name_from)
        from_element = temp_row[real_server_index]
        # 遍历该边缘节点连接的客户节点  查看是否有可以调整的客户节点
        for i in range(len(from_element)):
            from_point_value, from_point_name = from_element[i]
            if from_point_value == 0:
                continue
            point = points[from_point_name]

            # 寻找该客户节点连接到的所有可用的边缘节点   并按照95%位置带宽-当前带宽的方式计算其富余量，并进行从大到小的排序
            total_to_server_name = list(point.qos_server.keys())
            total_to_server_name.remove(server_name_from)
            for temp_i in range(len(total_to_server_name)):
                server_name_to = total_to_server_name[temp_i]
                server_point_temp = server_points[server_name_to]
                if num not in server_point_temp.dic95:
                    total_to_server_name[temp_i] = [0, server_name_to]
                    continue
                total_to_server_name[temp_i] = [server_point_temp.dic95[num] - server_point_temp.max_5th, server_name_to]
            total_to_server_name.sort(reverse=True)
            total_to_server_name = [j for (i, j) in total_to_server_name]

            # 遍历循环可用边缘节点  并进行调整  直到不可调整
            for server_name_to in total_to_server_name:
                server_index_to = server_points_ranking.index(server_name_to)
                to_element = temp_row[server_index_to]
                if to_element == None:
                    continue
                server_point_to = server_points[server_name_to]
                if server_point_to.max_5th_index == num:
                    continue
                max_push = server_point_to.dic95[num] - server_point_to.max_5th
                for j in range(len(to_element)):
                    to_point_value, to_point_name = to_element[j]
                    if to_point_name != from_point_name:
                        continue
                    if from_point_value <= max_push:
                        loss_value = from_point_value
                        from_point_value = 0
                        to_point_value += loss_value
                        server_point_to.dic95[num] -= loss_value
                        server_point_to.demand_sequence[num] -= loss_value
                        total_loss += loss_value
                    else:
                        loss_value = max_push
                        from_point_value -= loss_value
                        to_point_value += loss_value
                        server_point_to.dic95[num] -= loss_value
                        server_point_to.demand_sequence[num] -= loss_value
                        total_loss += loss_value
                    to_element[j][0] = to_point_value
                    break
                temp_row[server_index_to] = to_element
                if from_point_value == 0:
                    break
            from_element[i][0] = from_point_value
        # 更新相关参数   修改为调整后的结果
        temp_row[real_server_index] = from_element
        gradient_descent_matrix[num] = temp_row
        server_point_from.dic95[num] += total_loss
        server_point_from.demand_sequence[num] += total_loss
        server_point_from.max_5th += total_loss
        # loss等于0，代表已经暂时不可更新   可以暂时舍弃
        if total_loss == 0:
            # gd_done_list.append(server_name_from)
            if num in pass_point_dic:
                pass_point_dic[num].append([max_5th, server_name_from])
            else:
                pass_point_dic[num] = []
                pass_point_dic[num].append([max_5th, server_name_from])
        else:
            if num in pass_point_dic:
                for ele in pass_point_dic[num]:
                    server_points_ranking_list.append(ele)
                del pass_point_dic[num]
            # 更新该边缘节点的95%带宽值  并更新相应排序列表  进行下一轮循环
            server_point_from.renew_max_5th()
            server_points_ranking_list.append([server_point_from.max_5th, server_name_from])
            server_points_ranking_list.sort(reverse=True)
        # print(total_loss, "  gd success")

    # 依据上述梯度下降结果  确定最后的分配方案字符串
    for num in range(time_length):
        temp_row = gradient_descent_matrix[num]
        for index in range(len(server_points_ranking)):
            element = temp_row[index]
            if element == None:
                continue
            server_name = server_points_ranking[index]
            for used, name in element:
                if used == 0:
                    continue
                temp = "<" + server_name + "," + str(used) + ">"
                points[name].distribution_plan[num].append(temp)

    #############################################################################################################
    ########################         写入output文件           ####################################################
    #############################################################################################################
    with open(path4, "w") as f:
        for i in range(time_length-1):
            message = ""
            for point_name in name_list:
                message += point_name + ":"
                for plan in points[point_name].distribution_plan[i]:
                    message += plan + ","
                if message[-1] != ":":
                    message = message[:-1]
                message += "\n"
            f.write(message)

        message = ""
        i = time_length-1
        for point_name in name_list:
            message += point_name + ":"
            for plan in points[point_name].distribution_plan[i]:
                message += plan + ","
            if message[-1] != ":":
                message = message[:-1]
            message += "\n"
        message = message[:-1]
        f.write(message)
    #############################################################################################################
    #############################################################################################################
    #############################################################################################################

    print(time.time() - t)
