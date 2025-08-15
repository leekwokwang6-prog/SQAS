import utils
import argparse
import time
import tensorcircuit as tc
import numpy as np
import torch
import networkx as nx

tc.set_dtype("complex128")
tc.set_backend("tensorflow")

class Graph_property_cir:
    """
    有双量子门qubit限制的生成方式：按照available_edge选择双量子门可作用的位置
    """
    def __init__(self, adj_matrixs):
        self.adj_matrixs = adj_matrixs
        self.cirs_size = len(self.adj_matrixs)

    def average_path_length(self):
        self.cirs_size = len(self.adj_matrixs)
        cirs_length = []
        for i in range(0, self.cirs_size):
            G = nx.Graph(self.adj_matrixs[i])  # 将邻接矩阵转换为图
            avg_path_length = nx.average_shortest_path_length(G)
            cirs_length.append(np.array([avg_path_length]))
        return cirs_length

    # 定义一个函数来计算聚类系数
    def clustering_coefficient(self):
        """
        计算无向图的聚类系数
        参数：adjacency_matrix (numpy.ndarray): 表示无向图的邻接矩阵，其中 adjacency_matrix[i][j] 为1表示节点i和节点j之间有边，为0表示没有。
        返回值：float: 无向图的聚类系数
        注意：该函数假设输入的邻接矩阵是无向图的邻接矩阵。
        """
        # 获取节点数量
        self.cirs_size = len(self.adj_matrixs)
        cirs_clustering_coeffs = []
        for i in range(0, self.cirs_size):
            G = nx.Graph(self.adj_matrixs[i])  # 将邻接矩阵转换为图
            avg_clustering_coefficient = nx.average_clustering(G)
            cirs_clustering_coeffs.append(np.array([avg_clustering_coefficient]))
        return cirs_clustering_coeffs

    # 定义一个函数来计算度熵
    def degree_entropy(self):
        self.cirs_size = len(self.adj_matrixs)
        cirs_degree_entropy = []
        for i in range(0, self.cirs_size):
            num_nodes = len(self.adj_matrixs[i])  # 获取邻接矩阵的维度，即节点数量
            degrees = np.sum(self.adj_matrixs[i], axis=1)  # 对邻接矩阵的行进行求和
            # 计算度分布的概率分布
            degree_probabilities = []
            for i in range(num_nodes):
                degree_probabilities.append(degrees[i] / (np.sum(degrees) * 0.5))
            # 计算度分布熵，即 Shannon 熵
            degree_entropy_value = -sum(p * np.log2(p) for p in degree_probabilities) / num_nodes
            cirs_degree_entropy.append(np.array([degree_entropy_value]))
        return cirs_degree_entropy

    # 定义一个函数来计算楔数
    def wedge_count(self):
        """
        计算无向图中的楔数。
        参数：adjacency_matrix (numpy.ndarray): 无向图的邻接矩阵。
        返回：wedge_count (int): 楔数的数量。
        """
        self.cirs_size = len(self.adj_matrixs)
        cirs_wedge_count = []
        for i in range(0, self.cirs_size):
            num_nodes = len(self.adj_matrixs[i])
            wedge_count = 0
            degrees = np.sum(self.adj_matrixs[i], axis=1)  # 对邻接矩阵的行进行求和
            for i in range(num_nodes):
                if degrees[i] >= 2:
                    wedge_count += degrees[i] * (degrees[i] - 1) * 0.5
                else:
                    wedge_count += 0
            cirs_wedge_count.append(np.array([wedge_count]))
        return cirs_wedge_count

    # 定义一个函数来计算基尼指数
    def gini_index(self):
        self.cirs_size = len(self.adj_matrixs)
        cirs_gini_index = []
        for i in range(0, self.cirs_size):
            degrees = np.sum(self.adj_matrixs[i], axis=1)  # 获取节点的度数列表
            sorted_degrees = np.sort(degrees)  # 对度数列表进行排序
            num_nodes = len(degrees)  # 获取图中的节点数
            upper_part = 0
            lower_part = 0
            # 计算Gini指数
            for node in range(num_nodes):
                upper_part += 2 * (node + 1) * sorted_degrees[node]
                lower_part += num_nodes * sorted_degrees[node]
            gini_index = upper_part / lower_part - (num_nodes + 1) / num_nodes
            cirs_gini_index.append(np.array([gini_index]))
        return cirs_gini_index

    # 定义一个函数来计算弹性参数
    def resilience_parameter(self):
        self.cirs_size = len(self.adj_matrixs)
        cirs_resilience_parameter = []
        for i in range(0, self.cirs_size):
            degrees = np.sum(self.adj_matrixs[i], axis=1)  # 获取节点的度数列表
            num_nodes = len(self.adj_matrixs[i])  # 获取图中的节点数
            upper_part = 0
            # 计算resilience_parameter
            for node in range(num_nodes):
                upper_part += np.power(degrees[node], 2) / num_nodes
            resilience_parameter = upper_part / np.mean(degrees)
            cirs_resilience_parameter.append(np.array([resilience_parameter]))
        return cirs_resilience_parameter

    # 定义一个函数来计算紧密中心性
    def closeness_centrality(self):
        self.cirs_size = len(self.adj_matrixs)
        cirs_resilience_parameter = []
        for i in range(0, self.cirs_size):
            G = nx.Graph(self.adj_matrixs[i])  # 将邻接矩阵转换为图
            closeness_centrality = nx.closeness_centrality(G)  # 计算节点的接近中心性
            num_nodes = len(self.adj_matrixs[i])  # 获取图中的节点数
            avg_closeness_centrality = sum(closeness_centrality.values()) / num_nodes
            cirs_resilience_parameter.append(np.array([avg_closeness_centrality]))
        return cirs_resilience_parameter

    # 定义一个函数来计算偏心率
    def eccentricity(self):
        self.cirs_size = len(self.adj_matrixs)
        cirs_eccentricity = []
        for i in range(0, self.cirs_size):
            G = nx.Graph(self.adj_matrixs[i])  # 将邻接矩阵转换为图
            num_nodes = len(self.adj_matrixs[i])  # 获取图中的节点数
            eccentricities = nx.eccentricity(G)  # 计算节点的偏心率
            avg_eccentricities = sum(eccentricities.values()) / num_nodes
            # for value in eccentricities.values():
            cirs_eccentricity.append(np.array([avg_eccentricities]))
        return cirs_eccentricity

    # 定义一个函数来计算直径
    def diameter(self):
        self.cirs_size = len(self.adj_matrixs)
        cirs_diameter = []
        for i in range(0, self.cirs_size):
            G = nx.Graph(self.adj_matrixs[i])  # 将邻接矩阵转换为图
            eccentricities = nx.eccentricity(G)  # 计算节点的接近中心性
            max_eccentricities = max(eccentricities.values())
            cirs_diameter.append(np.array([max_eccentricities]))
        return cirs_diameter

    # 定义一个函数来计算直径
    def radius(self):
        self.cirs_size = len(self.adj_matrixs)
        cirs_radius = []
        for i in range(0, self.cirs_size):
            G = nx.Graph(self.adj_matrixs[i])  # 将邻接矩阵转换为图
            eccentricities = nx.eccentricity(G)  # 计算节点的接近中心性
            max_eccentricities = min(eccentricities.values())
            cirs_radius.append(np.array([max_eccentricities]))
        return cirs_radius


    def property_calculate(self):
        G_properties_list = []
        # 求邻接矩阵的平均路径长度值
        G_properties_1 = self.average_path_length()
        G_properties_list.append(G_properties_1)
        # 求邻接矩阵的聚类系数值
        G_properties_2 = self.clustering_coefficient()
        G_properties_list.append(G_properties_2)

        # 求邻接矩阵的度熵值
        G_properties_5 = self.degree_entropy()
        G_properties_list.append(G_properties_5)
        # 求邻接矩阵的契数值
        G_properties_6 = self.wedge_count()
        G_properties_list.append(G_properties_6)
        # 求邻接矩阵的基尼指数值
        G_properties_7 = self.gini_index()
        G_properties_list.append(G_properties_7)
        # 求邻接矩阵的弹性参数值
        G_properties_8 = self.resilience_parameter()
        G_properties_list.append(G_properties_8)
        # 求邻接矩阵的紧密中心性值
        G_properties_9 = self.closeness_centrality()
        G_properties_list.append(G_properties_9)
        # 求邻接矩阵的偏心率值
        G_properties_10 = self.eccentricity()
        G_properties_list.append(G_properties_10)
        # 求邻接矩阵的直径值
        G_properties_11 = self.diameter()
        G_properties_list.append(G_properties_11)
        # 求邻接矩阵的半径值
        G_properties_12 = self.radius()
        G_properties_list.append(G_properties_12)
        return G_properties_list

    def properties_cirs(self, G_properties_list):
        G_properties_cirs = [0] * self.cirs_size
        for j in range(0, self.cirs_size):
            temp = []
            for i in range(0, len(G_properties_list)):
                temp.append(G_properties_list[i][j][0])
            G_properties_cirs[j] = temp
        return G_properties_cirs

def find_optimal_route(pre_index_in_space,label,save_path):
    route = len(pre_index_in_space)
    temp_fo = 0
    for i in range(0, route):
        if label[pre_index_in_space[i]] < -8.47:
            print(f"总共{route}，在第{i+1}条找到了最好的量子线路")
            temp_fo = i + 1
            break
    # print(f"总共{route}，找不到最好的量子线路")
    utils.save_pkl(temp_fo , save_path + 'queries.pkl')
    utils.save_pkl(label[pre_index_in_space[temp_fo-1]],save_path +'energy_opt.pkl')
    return 0

def main(arg):
    start_time = time.time()
    np.random.seed(arg.seed)  # Set NumPy random seed
    torch.manual_seed(arg.seed)  # Set PyTorch random seed
    torch.cuda.manual_seed_all(arg.seed)  # Set PyTorch CUDA random seed
    torch.backends.cudnn.benchmark = False  # Disable auto optimization for reproducibility
    torch.backends.cudnn.deterministic = True

    save_datasets = f'./datasets/experiment_data/{arg.device}_{arg.task}/'

    '''Generate data'''
    dataset_types = ['train', 'val', 'test', 'test_eval']
    data_cir = {}
    for dataset_type in dataset_types:
        data_cir[dataset_type] = {}

    # Load data (training set)
    data_cir['train']['architecture'] = utils.load_pkl(save_datasets + f'training/run_{arg.seed}/samples.pkl')[:arg.numCir]
    data_cir['train']['matrix_cir'] = utils.load_pkl(save_datasets + f'training/run_{arg.seed}/adj_feat_matrix.pkl')[:arg.numCir]
    data_cir['train']['energy'] = utils.load_pkl(save_datasets + f'training/run_{arg.seed}/energy.pkl')[:arg.numCir]
    data_cir['train']['properties_list'] = utils.load_pkl(save_datasets + f'training/run_{arg.seed}/properties_list.pkl')[:arg.numCir]

    # Load data (test set)
    data_cir['test']['architecture'] = utils.load_pkl(save_datasets + f'search_pool/run_{arg.seed}/samples.pkl')[:arg.numCir_unlabel]
    data_cir['test']['matrix_cir'] = utils.load_pkl(save_datasets + f'search_pool/run_{arg.seed}/adj_feat_matrix.pkl')[:arg.numCir_unlabel]
    data_cir['test']['properties_list'] = utils.load_pkl(save_datasets + f'search_pool/run_{arg.seed}/properties_list.pkl')[:arg.numCir_unlabel]

    """变无向图"""
    matrix_cir = data_cir['train']['matrix_cir']
    for i in range(0, len(matrix_cir)):
        data_cir['train']['matrix_cir'][i][0] = np.logical_or(matrix_cir[i][0], matrix_cir[i][0].T).astype(int)
    adj_train = [matrix_cir[i][0] for i in range(len(matrix_cir))]

    matrix_cir = data_cir['test']['matrix_cir']
    for i in range(0, len(matrix_cir)):
        data_cir['test']['matrix_cir'][i][0] = np.logical_or(matrix_cir[i][0], matrix_cir[i][0].T).astype(int)
    adj_test = [matrix_cir[i][0] for i in range(len(matrix_cir))]

    print("求图属性")
    """求图属性"""
    GP = Graph_property_cir(adj_matrixs=adj_train)
    G_properties_list = GP.property_calculate()
    properties_list = GP.properties_cirs(G_properties_list)
    data_cir['train']['properties_list'] = properties_list
    utils.save_pkl(properties_list, f'datasets/experiment_data/{arg.device}_{arg.task}/training/run_{arg.seed}/properties_list.pkl')


    GP = Graph_property_cir(adj_matrixs=adj_test)
    G_properties_list = GP.property_calculate()
    properties_list = GP.properties_cirs(G_properties_list)
    data_cir['test']['properties_list'] = properties_list
    utils.save_pkl(properties_list, f'datasets/experiment_data/{arg.device}_{arg.task}/search_pool/run_{arg.seed}/properties_list.pkl')

    return


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--task', type=str, default='TFIM_8', help='see configs for available tasks')
    parser.add_argument('--device', type=str, default='grid_16q', help='')
    parser.add_argument('--method', type=str, default='Teacher-Student', help='Search method')
    parser.add_argument("--numCir", type=int, default=300, help="Number of labeled dataset")
    parser.add_argument("--numCir_unlabel", type=int, default=100000, help="Number of unlabeled dataset")

    args = parser.parse_args()  # 解析用户提供的命令行参数
    main(args)
    end_time = time.time()
