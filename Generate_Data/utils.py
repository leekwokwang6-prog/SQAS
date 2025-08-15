import os
import pickle
import numpy as np
import networkx as nx
import pennylane as qml
import matplotlib.pyplot as plt

def make_dir(dir_file):
    '''
    :param dir_file: e.g., 'result/1.txt'
    '''
    dir = os.path.dirname(dir_file)    # 'result'
    os.makedirs(dir, exist_ok=True)

def save_pkl(data, dir_file):
    '''
    :param dir_file: e.g., 'result/1.txt'
    '''
    make_dir(dir_file)
    f = open(dir_file, 'wb')
    pickle.dump(data, f)
    f.close()

def load_pkl(dir_file):
    f = open(dir_file, 'rb')
    data = pickle.load(f)
    f.close()
    return data

def make_it_unique(circuit, num_qubit):
    '''
    circuit: a circuit in list format, e.g.:  [['rz', 0], ['cz', 2, 3], ['ry', 1]]
    return: list: a list of rearranged gate sequence where all gates are moved to the far left,
            int: circuit depth
            int: number of gates
            int: number of 2-qubit gates
    '''
    bitmap = []
    for _ in range(0, num_qubit):
        bitmap.append([])

    num_1_qb_gate, num_2_qb_gate = 0, 0
    for gate in circuit:
        if len(gate) == 2:  # 1-qubit gate
            bitmap[gate[1]].append(gate)
            num_1_qb_gate = num_1_qb_gate + 1
        else:  # 2-qubit gate
            if len(bitmap[gate[1]]) >= len(bitmap[gate[2]]):
                qb_long, qb_short = gate[1], gate[2]
            else:
                qb_long, qb_short = gate[2], gate[1]
            while len(bitmap[qb_short]) < len(bitmap[qb_long]):
                bitmap[qb_short].append(None)

            if gate[1] > gate[2]:
                qb_low, qb_high = gate[2], gate[1]
            else:
                qb_low, qb_high = gate[1], gate[2]
            bitmap[qb_low].append(gate)
            bitmap[qb_high].append(None)
            num_2_qb_gate = num_2_qb_gate + 1

    num_bit = []
    for i in range(0, num_qubit):
        num_bit.append(len(bitmap[i]))
    depth = max(num_bit)    # depth of the circuit

    uni_cir = []
    for i in range(depth):
        for j in range(num_qubit):
            if num_bit[j] > i:
                if bitmap[j][i] is not None:
                    uni_cir.append(bitmap[j][i])

    return uni_cir.copy(), depth, num_1_qb_gate+num_2_qb_gate, num_2_qb_gate


def circuit_list_to_adj(circuit, num_qubit, gate_type_list, require_feat_matrix=False):
    '''
    produce the adjacency matrix and feat matrix based on the DAG of a circuit
    :param circuit: list, e.g., [['rz', 0], ['ry', 2], ['cz', 1, 2]], processed by make_it_unique
    :param num_qubit: int
    :param gate_type_list: list, e.g., ['rz', 'ry', 'cz']
    :param require_feat_matrix: bool, return the feat matrix or not
    :return: ndarrray, adjacency matrix
             ndarrray, feat matrix
    '''

    graph = nx.DiGraph()

    # add nodes to graph
    graph.add_node('start')
    for j in range(1, len(circuit)+1):
        graph.add_node(j)
    graph.add_node('end')

    # add edges to graph
    last = ['start' for _ in range(num_qubit)]
    for k, gate in enumerate(circuit):
        if len(gate) == 2:  # 1-qubit gate
            graph.add_edge(last[gate[1]], k+1)
            last[gate[1]] = k+1
        else:  # 2-qubit gate
            graph.add_edge(last[gate[1]], k+1)
            graph.add_edge(last[gate[2]], k+1)
            last[gate[1]] = k+1
            last[gate[2]] = k+1
    for k in last:
        graph.add_edge(k, 'end')

    # nx.draw_networkx(graph)
    # plt.show()
    # plt.savefig('DAG.png')

    # get the adjacency matrix
    adj_matrix = nx.adjacency_matrix(graph).todense()

    if require_feat_matrix:
        feat_matrix = []
        for node in graph.nodes:
            t1 = [0 for _ in range(len(gate_type_list) + 2)]
            if node == 'start':
                t1[0] = 1
                t2 = [1 for _ in range(num_qubit)]
            elif node == 'end':
                t1[-1] = 1
                t2 = [1 for _ in range(num_qubit)]
            else:
                t1[gate_type_list.index(circuit[node - 1][0]) + 1] = 1
                t2 = [0 for _ in range(num_qubit)]
                t2[circuit[node-1][1]] = 1
                if len(circuit[node-1]) == 3:
                    t2[circuit[node-1][2]] = 1
            t1.extend(t2)
            feat_matrix.append(t1)
        feat_matrix = np.array(feat_matrix)
        return adj_matrix, feat_matrix
    else:
        return adj_matrix

def draw_circuit(cir_lst, num_qubit, dir_file):
    '''
    :param cir_lst: list, e.g., [['rz',0], ['ry',1], ['cz',0,1]]
    :param num_qubit: int, number of qubits
    :param dir_file: str, e.g., 'result/cir.png'
    '''
    dev = qml.device('default.qubit', wires=num_qubit)

    @qml.qnode(dev)
    def circuit(cir):
        for gate in cir:
            if gate[0] == 'cx':
                qml.CNOT(wires=gate[1:])
            elif gate[0] == 'cz':
                qml.CZ(wires=gate[1:])
            elif gate[0] == 'rx':
                qml.RX(0, wires=gate[1])
            elif gate[0] == 'ry':
                qml.RY(0, wires=gate[1])
            elif gate[0] == 'rz':
                qml.RZ(0, wires=gate[1])
            elif gate[0] == 'x':
                qml.PauliX(wires=gate[1])
            elif gate[0] == 'sx':
                qml.SX(wires=gate[1])
            else:
                print(f'undefined gate {gate[0]} in drawing circuit')
                exit(123)
        return qml.state()

    qml.draw_mpl(circuit)(cir_lst)
    make_dir(dir_file)
    plt.savefig(dir_file)
    plt.close()

def levenshtein_distance(s1, s2):
    '''
    :param s1: list, a sequence
    :param s2: list, another sequence
    :return: int, Levenshtein distance between two sequences
    '''
    # 创建一个 (m+1)x(n+1) 的矩阵
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化边界条件
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # 填充矩阵
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,    # 删除
                           dp[i][j - 1] + 1,    # 插入
                           dp[i - 1][j - 1] + cost)  # 替换

    return dp[m][n]


