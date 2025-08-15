
task_configs = {
    'Heisenberg_8':
        {'qubit': 8,
         'layer': 10,
         'Hamiltonian': {'pbc': True, 'hzz': 1, 'hxx': 1, 'hyy': 1, 'hx': 0, 'hy': 0, 'hz': 1, 'sparse': False},
         },
    'TFIM_8':
        {'qubit': 8,
         'layer': 10,
         'Hamiltonian': {'pbc': True, 'hzz': 1, 'hxx': 0, 'hyy': 0, 'hx': 1, 'hy': 0, 'hz': 0, 'sparse': False},
         },
}

device = {'ishape': # the I-shaped 7-qubit local structure in IBM's QPU
              {'n_qubit': 7,
               'gate_set': ['cz', 'rz', 'ry'],
               'connectivity': [[0, 1], [1, 2], [1, 3], [3, 5], [4, 5], [5, 6]],
               'layout': [[[0, 1], [3, 5]],
                          [[1, 2], [3, 5]],
                          [[1, 3], [4, 5]],
                          [[1, 3], [5, 6]],
                          [[0, 1], [4, 5]],
                          [[1, 2], [5, 6]],
                          [[0, 1], [5, 6]],
                          [[1, 2], [4, 5]]]
              },
          'grid_7q': # a grid topology with 7 qubits
              {'n_qubit': 7,
               'gate_set': ['cx', 'rx', 'ry'],
               'connectivity': [[0, 1], [0, 4], [1, 2], [1, 5], [2, 3], [2, 6], [4, 5], [5, 6]],
               'layout': [[[0, 4], [1, 5], [2, 6]],
                          [[0, 4], [1, 5], [2, 3]],
                          [[0, 4], [1, 2], [5, 6]],
                          [[0, 4], [2, 3], [5, 6]],
                          [[0, 1], [2, 6], [4, 5]],
                          [[0, 1], [2, 3], [4, 5]],
                          [[0, 1], [2, 3], [5, 6]],
                          [[1, 2], [4, 5]]]
              },
          'grid_8q': # a grid topology with 8 qubits
              {'n_qubit': 8,
               'gate_set': ['cx', 'rx', 'ry'],
               'connectivity': [[0, 1], [0, 4], [1, 2], [1, 5], [2, 3],
                                [2, 6], [3, 7], [4, 5], [5, 6], [6, 7]],
               'layout': [[[0, 1], [2, 3], [4, 5], [6, 7]],
                          [[0, 4], [1, 5], [2, 6], [3, 7]],
                          [[0, 4], [1, 2], [3, 7], [5, 6]],
                          [[0, 1], [2, 3], [5, 6]],
                          [[1, 2], [4, 5], [6, 7]],
                          [[0, 4], [1, 5], [2, 3], [6, 7]],
                          [[0, 1], [2, 6], [3, 7], [4, 5]],
                          [[0, 4], [1, 2], [6, 7]],
                          [[0, 4], [2, 3], [5, 6]],
                          [[0, 1], [3, 7], [5, 6]],
                          [[1, 2], [3, 7], [4, 5]]]
              },
          'grid_16q': # a grid topology with 8 qubits
              {'n_qubit': 16,
               'gate_set': ['cz', 'rz', 'ry'],
               'connectivity': [[0, 1], [0, 4], [1, 2], [2, 3], [2, 6], [3, 7], [4, 5], [4, 8], [5, 6], [5, 9], [6, 7],
                                [7, 11], [8, 9], [8, 12], [9, 10], [9, 13], [10, 11], [12, 13], [13, 14]],
              }
}

opt_cfg = {'optimizer': ['Adagrad', 'Adam', 'Nadam', 'RMSprop', 'SGD'],
           'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
           'batch_size': [32, 64, 128, 256]
}

def get_input_dim(device_name):
    '''
    :return: dim of node feature
    '''
    input_dim = device[device_name]['n_qubit'] + len(device[device_name]['gate_set']) + 2
    return input_dim

def get_gcn_setting(device_name):
    gcn_setting = {"num_layer": 5,
                   "input_dim": get_input_dim(device_name),
                   "output_dim": 16,
                   "hidden_dim": 128}
    return gcn_setting

def get_gmae_setting(device_name):
    '''
    latent_feature_dim must equal gcn's output_dim
    '''
    gmae_setting = {"in_feature_dim": get_input_dim(device_name),
                    "latent_feature_dim": 16}
    return gmae_setting

def get_predictor_setting():
    '''
    latent_feature_dim must equal gcn's output_dim
    '''
    opt_cfg_dim = len(opt_cfg['optimizer']) \
                  + len(opt_cfg['learning_rate'])

    predictor_setting = {"latent_feat_dim": 16,
                         "opt_cfg_dim": opt_cfg_dim,
                         'hidden_dim': 128,
                         'num_layers': 2,
                         'dropout': 0.7}
    return predictor_setting




