import numpy as np

def build_dataset(adj_feat_matrix, test_ratio):
    '''
    Split into training and testing sets, padded with zeros
    :param adj_feat_matrix: list，each element is [adj_matrix, feat_matrix], two matrices are ndarray
    :param test_ratio: float, the ratio of test samples, e.g., 0.2
    :return: X_adj_train: ndarray, (num_sample, num_node, num_node)
             X_feat_train: ndarray, (num_sample, num_node, feat_dim)
             X_adj_test: ndarray, (num_sample, num_node, num_node)
             X_feat_test: ndarray, (num_sample, num_node, feat_dim)
    '''
    # Pad with zeros to equalize the number of nodes
    adj, feat = pad_with_zero(adj_feat_matrix)

    return shuffle(adj, feat, test_ratio)

def shuffle(X, Y, test_ratio):
    '''
    :param X: ndarray, a batch of data
    :param Y: ndarray, a batch of data
    :param test_ratio: ratio of the test set
    :return: ndarray, X_train, Y_train, X_test, Y_test
    '''
    # randomly split the dataset into training and testing sets
    num_sample = X.shape[0]
    num_test_sample = int(num_sample * test_ratio)
    num_train_sample = num_sample - num_test_sample
    indices = np.random.permutation(num_sample)
    train_idx_list = indices[0:int(num_train_sample)]
    test_idx_list = indices[int(num_train_sample):]
    X_train, Y_train = X[train_idx_list], Y[train_idx_list]
    X_test, Y_test = X[test_idx_list], Y[test_idx_list]

    return X_train, Y_train, X_test, Y_test


def pad_with_zero(adj_feat_matrix):
    """
    Pad adj_matrix and feat_matrix in the adj_feat_matrix list to the same shape
    """
    # Step 1: Find the maximum shape for adj_matrix and feat_matrix separately
    max_num_node = max(matrix_pair[0].shape[0] for matrix_pair in adj_feat_matrix)
    feat_dim = adj_feat_matrix[0][1].shape[1]

    # Step 2: Pad each adj_matrix and gate_matrix
    padded_adj_matrices = []
    padded_feat_matrices = []

    for adj_matrix, feat_matrix in adj_feat_matrix:
        padded_adj = pad_matrix(adj_matrix, (max_num_node, max_num_node))
        padded_feat = pad_matrix(feat_matrix, (max_num_node, feat_dim))

        padded_adj_matrices.append(padded_adj)
        padded_feat_matrices.append(padded_feat)

    adj = np.array(padded_adj_matrices)
    feat = np.array(padded_feat_matrices)

    return adj, feat

def pad_matrix(matrix, target_shape):
    """
    Pad the input matrix to the target shape with zeros.
    """
    padded_matrix = np.zeros(target_shape, dtype=matrix.dtype)
    padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
    return padded_matrix




