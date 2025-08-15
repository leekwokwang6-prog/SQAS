import argparse
import utils
import time
import torch
from model.mlp_model_Dropout import MLP
from TFIM_task_paralell import VqeTrainerNew
import os
import numpy as np

def normalize_features(x):
    '''
    Normalize each sample (row) by subtracting the mean and dividing by the standard deviation of its features.
    :param x: numpy array, shape (batch_size, -1)
    :return: numpy array, shape (batch_size, -1)
    '''
    mean = np.mean(x, axis=0, keepdims=True)  # Compute the mean of each row
    std = np.std(x, axis=0, keepdims=True) + 1e-8  # Compute the standard deviation of each row and add a small value to prevent division by zero
    return (x - mean) / std

def qas(args, device_name, task_name, run_id):
    start_time_qas = time.time()

    save_datasets = f'{args.save_datasets}{device_name}_{task_name}/'
    samples = utils.load_pkl(save_datasets + f'search_pool/run_{run_id}/samples.pkl')
    properties_list = utils.load_pkl(save_datasets + f'search_pool/run_{run_id}/properties_list.pkl')

    properties_list = normalize_features(np.array(properties_list))
    properties_list = torch.Tensor(properties_list)

    model = MLP(2, 10, 128, 1, args.dropout)
    model.load_state_dict(torch.load(f'{args.exp_dir}{args.model_path}'))

    start_time_infer = time.time()
    # predict
    model.eval()
    with torch.no_grad():
        pred_value = model(properties_list)
    pred_value = pred_value.squeeze().tolist()
    end_time_infer = time.time()

    save_result_data = f"{args.exp_dir}run_{args.run_id}/"
    if not os.path.exists(save_result_data):
        os.makedirs(save_result_data)

    # rank
    ranking = sorted(range(len(pred_value)), key=lambda i: pred_value[i], reverse=False)
    utils.save_pkl(ranking, save_result_data +f'ranking.pkl')


    qubit = args.n_qubit
    noise_param = None
    if args.noise:
        noise_param = {'two_qubit_channel_depolarizing_p': args.two_qubit_depolarizing_p,
                       'single_qubit_channel_depolarizing_p': args.single_qubit_depolarizing_p,
                       'bit_flip_p': args.bit_flip_p}
    hamiltonian = {'pbc': True, 'hzz': 1, 'hxx': 0, 'hyy': 0, 'hx': 1, 'hy': 0, 'hz': 0, 'sparse': False}
    trainer = VqeTrainerNew(n_cir_parallel=10, n_runs=10, max_iteration=2000, n_qubit=qubit, hamiltonian=hamiltonian,
                            noise_param=noise_param)

    samples_eval = []
    pred_value_rank = []
    for i in range(0, 100):     # train TOP-150 circuits
        print(f'circuit id: {i}')
        samples_eval.append(samples[ranking[i]])
        pred_value_rank.append(pred_value[ranking[i]])

    utils.save_pkl(pred_value_rank, save_result_data + f'pred_value_rank.pkl')
    utils.save_pkl(samples_eval, save_result_data + f'samples_rank.pkl')

    energy, param, energy_epoch, duration = trainer.batch_train_parallel(samples_eval, device_name, task_name, run_id)
    end_time_qas = time.time()


    if not os.path.exists(save_result_data):
        os.makedirs(save_result_data)
    utils.save_pkl(energy, save_result_data + f'energy.pkl')
    utils.save_pkl(param, save_result_data + f'param.pkl')
    utils.save_pkl(energy_epoch, save_result_data + f'energy_epoch.pkl')
    utils.save_pkl(end_time_qas - start_time_qas, save_result_data + f'qas_time.pkl')
    return end_time_infer -  start_time_infer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--run_id", type=int, default=0, help='')
    parser.add_argument("--noise", type=bool, default=True, help="Whether to consider noise")
    parser.add_argument("--two_qubit_depolarizing_p", type=float, default=0.01, help="Two-qubit depolarizing noise probability")
    parser.add_argument("--single_qubit_depolarizing_p", type=float, default=0.001, help="Single-qubit depolarizing noise probability")
    parser.add_argument("--bit_flip_p", type=float, default=0.01, help="Bit-flip noise probability")
    parser.add_argument("--n_qubit", type=int, default=8, help="Number of qubits")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument('--task', type=str, default='TFIM_8', help='See configs for available tasks')
    parser.add_argument('--device', type=str, default='grid_16q', help='')
    parser.add_argument('--save_datasets', type=str, default=f'./datasets/experiment_data/', help='')
    parser.add_argument("--exp_dir", type=str, default=f'saved_models_EMA_TFIM_728/', help='')
    parser.add_argument("--model_path", type=str, default=f'predictor_UACS_seed0.pth', help='')
    args = parser.parse_args()

    start_time = time.time()
    device_name = args.device
    task_name = args.task
    run_id = args.run_id
    qas(args, device_name, task_name, run_id)
    end_time = time.time()
    duration = end_time - start_time
