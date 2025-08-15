import utils
import os
import argparse
import time
from model.mlp_model_Dropout import MLP
import tensorcircuit as tc
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import spearmanr
from model import ramps
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
tc.set_dtype("complex128")
tc.set_backend("tensorflow")


def normalize_features(x):
    '''
    Normalize each sample (row), by subtracting mean and dividing by std of its features.
    :param x: numpy array, shape (batch_size, -1)
    :return: numpy array, shape (batch_size, -1)
    '''
    mean = np.mean(x, axis=0, keepdims=True)  # Compute mean of each row
    std = np.std(x, axis=0, keepdims=True) + 1e-8  # Compute std of each row and add a small value to avoid division by zero
    return (x - mean) / std

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more accurate

    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def confident_choose(arg, properties_train_unlabeled, GP_predictor, properties_train, y_train,
                     all_y_trains, all_means_trains, all_stds_trains, topkIdexs):
    """
    Perform infer_N Monte Carlo inferences on all unlabeled data,
    compute mean and std for each sample,
    then select top_k samples with smallest std as pseudo-labeled samples.
    Returns a new DataLoader including these top_k pseudo-labeled samples.
    """
    # Ensure model is in train mode to enable Dropout
    GP_predictor.train()

    with torch.no_grad():
        all_outputs = []
        all_outputs_train = []
        for _ in range(arg.infer_N):
            out = GP_predictor(properties_train_unlabeled)        # [N_unlabeled]
            all_outputs.append(out.detach().cpu())

            out_train = GP_predictor(properties_train)
            all_outputs_train.append(out_train.detach().cpu())
        all_outputs = torch.stack(all_outputs, dim=0)
        all_outputs_train = torch.stack(all_outputs_train, dim=0)

        # Compute std and mean for each sample
        stds = all_outputs.std(dim=0)   # [N_unlabeled]
        means = all_outputs.mean(dim=0) # [N_unlabeled]

        stds_train = all_outputs_train.std(dim=0)  # [N_unlabeled]
        means_train = all_outputs_train.mean(dim=0)  # [N_unlabeled]

        # Convert to CPU tensors
        y_train_arr = y_train.cpu()
        means_train_arr = means_train.cpu()
        stds_train_arr = stds_train.cpu()

        # —— New: append to external lists ——
        all_y_trains.append(y_train_arr)
        all_means_trains.append(means_train_arr)
        all_stds_trains.append(stds_train_arr)

        _, topk_idxs = torch.topk(-stds, arg.top_k)  # Negate to get smallest values
        topkIdexs.append(topk_idxs)

        selected_props = properties_train_unlabeled[topk_idxs].cpu()   # [top_k, ...]
        selected_pseudos = means[topk_idxs]                             # [top_k]

    # Concatenate with original labeled data
    properties_train_new = torch.cat([properties_train.cpu(), selected_props], dim=0)
    y_train_new = torch.cat([y_train.cpu(), selected_pseudos], dim=0)

    # Create new training dataset
    train_data = TensorDataset(properties_train_new, y_train_new)
    return train_data, properties_train_new, y_train_new



def PQAS_GP_search_test(data_cir, arg):
    # Load data
    properties_train, y_train = data_cir['train']['properties_list'], data_cir['train']['energy']
    properties_train_unlabeled = data_cir['test']['properties_list']  # Unlabeled samples
    properties_eval_unlabeled = data_cir['test_eval']['properties_list']  # Unlabeled samples
    properties_test, y_test = data_cir['val']['properties_list'], data_cir['val']['energy']

    if arg.properties_normal:
        properties_train = normalize_features(np.array(properties_train))  # Normalize training set
        properties_train_unlabeled = normalize_features(np.array(properties_train_unlabeled))
        properties_eval_unlabeled = normalize_features(np.array(properties_eval_unlabeled))
        properties_test = normalize_features(np.array(properties_test))  # Normalize test set

    # Convert to tensors
    properties_train = torch.Tensor(properties_train)
    properties_train_unlabeled = torch.Tensor(properties_train_unlabeled)
    properties_eval_unlabeled = torch.Tensor(properties_eval_unlabeled)
    properties_test = torch.Tensor(properties_test)

    y_train = normalize_features(np.array(y_train))
    y_test = normalize_features(np.array(y_test))

    y_train = torch.Tensor(y_train)
    y_test = torch.Tensor(y_test)

    # Create training DataLoaders
    train_data_warm = TensorDataset(properties_train, y_train)
    train_warmup_loader = DataLoader(train_data_warm, batch_size=arg.bs, shuffle=True, drop_last=False)

    train_data_labeled = TensorDataset(properties_train, y_train)
    train_labeled_loader = DataLoader(train_data_labeled, batch_size=arg.bs, shuffle=True, drop_last=False)

    train_data_unlabeled = TensorDataset(properties_train_unlabeled)
    train_unlabeled_loader = DataLoader(train_data_unlabeled, batch_size=arg.bs, shuffle=True, drop_last=False)

    # Initialize predictors
    GP_predictor = MLP(arg.loop, 10, arg.hidden, 1, arg.dropout)  # Assume MLP is predefined
    GP_predictor_TS = MLP(arg.loop, 10, arg.hidden, 1, arg.dropout)

    optimizer = torch.optim.Adam(GP_predictor.parameters(), lr=arg.lr_for_predictor)
    loss_func = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    total_loss_train = []
    loss_EVAL = [] # loss parts for each epoch
    loss_TRAIN = []

    loss_std_train_sup = [] # supervised loss only, student
    loss_std_eval_sup = []

    loss_ema_train_sup = [] # supervised loss only, teacher
    loss_ema_eval_sup = []

    spearman_coeffs = []
    spearman_coeffs_evl = []
    loss_pseudo = []

    # At the top of your script (or before main training function):
    all_y_trains = []
    all_means_trains = []
    all_stds_trains = []
    topkIdexs =[]
    pseudo_time = []

    global_step = 0

    start_time_pseudo = time.time()
    # 1. Warmup training
    for epoch in range(arg.warmup_epochs):
        GP_predictor.train()
        train_loss = []

        for step, (b_properties, b_y) in enumerate(train_warmup_loader):

            optimizer.zero_grad()
            b_properties, b_y = b_properties, b_y
            # Forward pass
            forward = GP_predictor(b_properties)  # Output of main model
            loss = loss_func(forward, b_y)

            # Backward pass
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        scheduler.step(np.average(train_loss))
        GP_predictor.eval()
        with torch.no_grad():
            # Validation set
            forward = GP_predictor(properties_test)
            forward_train = GP_predictor(properties_train)

            loss_train = loss_func(forward_train, y_train)
            loss_eval = loss_func(forward, y_test)

            loss_TRAIN.append(loss_train)
            loss_EVAL.append(loss_eval)

            loss_pseudo.append(loss_train)
            loss_std_train_sup.append(loss_train)
            loss_std_eval_sup.append(loss_eval)
            loss_ema_train_sup.append(loss_train)
            loss_ema_eval_sup.append(loss_eval)

        coeff_train, _ = spearmanr(forward_train.detach().numpy(), y_train.detach().numpy())
        coeff_evl, _ = spearmanr(forward.detach().numpy(), y_test.detach().numpy())
        spearman_coeffs.append(coeff_train)
        spearman_coeffs_evl.append(coeff_evl)

        # Record average training loss for current epoch
        total_loss_train.append(np.mean(train_loss))


    train_data, properties_train_new, y_train_new = confident_choose(arg, properties_train_unlabeled, GP_predictor,
                                                                         properties_train, y_train,
                                                                         all_y_trains, all_means_trains, all_stds_trains, topkIdexs)

    optimizer_TS = torch.optim.Adam(GP_predictor.parameters(), lr=arg.lr_for_predictor)
    scheduler_TS = ReduceLROnPlateau(optimizer_TS, 'min', factor=0.1, patience=10, verbose=True)

    for epoch in range(arg.pre_epochs):
        GP_predictor.train()
        train_loss = []

        # Update dataset (unlabeled part) each epoch
        train_loader = DataLoader(train_data, batch_size=arg.bs, shuffle=True, drop_last=False)

        for step, (b_properties, b_y) in enumerate(train_loader):

            optimizer_TS.zero_grad()
            # Take one batch
            b_properties, b_y = b_properties, b_y  # 64 labeled samples

            # Forward pass
            forward = GP_predictor(b_properties)
            # Supervised loss
            loss_sup = loss_func(forward, b_y)
            loss = loss_sup

            # Backward pass
            loss.backward()
            optimizer_TS.step()
            train_loss.append(loss.item())
            global_step += 1

        scheduler_TS.step(np.average(train_loss))
        GP_predictor.eval()
        with torch.no_grad():
            # Validation set
            forward_train = GP_predictor(properties_train)
            loss_sup_train = loss_func(forward_train, y_train)
            loss_TRAIN.append(loss_sup_train)

            # Include pseudo-label loss
            forward_train_pseudo = GP_predictor(properties_train_new)
            loss_sup_train_pseudo = loss_func(forward_train_pseudo, y_train_new)
            loss_pseudo.append(loss_sup_train_pseudo)

            # Validation loss
            forward_eval = GP_predictor(properties_test)
            loss_sup_eval = loss_func(forward_eval, y_test)
            loss_EVAL.append(loss_sup_eval)

        coeff_train, _ = spearmanr(forward_train.detach().numpy(), y_train.detach().numpy())
        coeff_evl, _ = spearmanr(forward_eval.detach().numpy(), y_test.detach().numpy())
        spearman_coeffs.append(coeff_train)
        spearman_coeffs_evl.append(coeff_evl)

        # Record average training loss for current epoch
        total_loss_train.append(np.mean(train_loss))
        train_data, properties_train_new, y_train_new = confident_choose(arg, properties_train_unlabeled, GP_predictor,
                                                                         properties_train, y_train,
                                                                         all_y_trains, all_means_trains, all_stds_trains,topkIdexs)

    GP_predictor.eval()
    with torch.no_grad():
        pred_test = GP_predictor(properties_test).cpu().numpy()
        pred_train = GP_predictor(properties_train).cpu().numpy()

    end_time_pseudo = time.time()
    pseudo_time.append(end_time_pseudo - start_time_pseudo)

    metrics_path_file = f'saved_models_UACS_{arg.device}_{arg.task}'
    if not os.path.exists(metrics_path_file):
        os.makedirs(metrics_path_file)

    GP_predictor.eval()
    # Save GP_predictor state_dict here
    save_path = f"{metrics_path_file}/predictor_UACS_seed{arg.seed}.pth"
    # Ensure folder exists before saving
    torch.save(GP_predictor.state_dict(), save_path)
    print(f"    >>> Parameters saved to: {save_path}")

    # Compute ranking
    rank = np.argsort(np.array(pred_test))
    return rank, pred_test, pred_train



def main(arg):
    start_time = time.time()
    np.random.seed(arg.seed)  # Set NumPy RNG seed
    torch.manual_seed(arg.seed)  # Set PyTorch CPU RNG seed
    torch.cuda.manual_seed_all(arg.seed)  # Set PyTorch CUDA RNG seed
    torch.backends.cudnn.benchmark = False  # Disable auto-optimization for reproducibility
    torch.backends.cudnn.deterministic = True

    save_datasets = f'./datasets/experiment_data/{arg.device}_{arg.task}/'

    '''Generate data'''
    dataset_types = ['train', 'val', 'test', 'test_eval']
    data_cir = {}
    for dataset_type in dataset_types:
        data_cir[dataset_type] = {}

    # Load training data
    data_cir['train']['architecture'] = utils.load_pkl(save_datasets + f'training/run_{arg.seed}/samples.pkl')[:arg.numCir]
    data_cir['train']['matrix_cir'] = utils.load_pkl(save_datasets + f'training/run_{arg.seed}/adj_feat_matrix.pkl')[:arg.numCir]
    data_cir['train']['energy'] = utils.load_pkl(save_datasets + f'training/run_{arg.seed}/energy.pkl')[:arg.numCir]
    data_cir['train']['properties_list'] = utils.load_pkl(save_datasets + f'training/run_{arg.seed}/properties_list.pkl')[:arg.numCir]

    # Load validation data
    data_cir['val']['architecture'] = utils.load_pkl(save_datasets + f'training/run_{(arg.seed + 1) % 5}/samples.pkl')[:arg.numCir]
    data_cir['val']['matrix_cir'] = utils.load_pkl(save_datasets + f'training/run_{(arg.seed + 1) % 5}/adj_feat_matrix.pkl')[:arg.numCir]
    data_cir['val']['energy'] = utils.load_pkl(save_datasets + f'training/run_{(arg.seed + 1) % 5}/energy.pkl')[:arg.numCir]
    data_cir['val']['properties_list'] = utils.load_pkl(save_datasets + f'training/run_{(arg.seed + 1) % 5}/properties_list.pkl')[:arg.numCir]

    # Load test data
    data_cir['test']['architecture'] = utils.load_pkl(save_datasets + f'search_pool/run_{arg.seed}/samples.pkl')[:arg.numCir_unlabel]
    data_cir['test']['matrix_cir'] = utils.load_pkl(save_datasets + f'search_pool/run_{arg.seed}/adj_feat_matrix.pkl')[:arg.numCir_unlabel]
    data_cir['test']['properties_list'] = utils.load_pkl(save_datasets + f'search_pool/run_{arg.seed}/properties_list.pkl')[:arg.numCir_unlabel]

    # Load evaluation data
    data_cir['test_eval']['architecture'] = utils.load_pkl(save_datasets + f'search_pool/run_{(arg.seed + 1) % 5}/samples.pkl')[:arg.numCir_unlabel]
    data_cir['test_eval']['matrix_cir'] = utils.load_pkl(save_datasets + f'search_pool/run_{(arg.seed + 1) % 5}/adj_feat_matrix.pkl')[:arg.numCir_unlabel]
    data_cir['test_eval']['properties_list'] = utils.load_pkl(save_datasets + f'search_pool/run_{(arg.seed + 1) % 5}/properties_list.pkl')[:arg.numCir_unlabel]

    rank, pred, pred_train = PQAS_GP_search_test(data_cir, arg)

    """Performance evaluation"""
    return


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument('--task', type=str, default='TFIM_8', help='see congigs for available tasks')
    parser.add_argument('--device', type=str, default='grid_16q', help='')
    parser.add_argument("--dropout", type=float, default=0.5, help='dropout')
    parser.add_argument("--loop", type=int, default=2, help='dropout')
    parser.add_argument("--hidden", type=int, default=64, help='dropout')
    parser.add_argument('--normal_type', type=str, default='normal', help='search method')
    parser.add_argument('--properties_normal', type=int, default=1, help='whether to load circuit files from a specified location')

    parser.add_argument("--warmup_epochs", type=int, default=300, help="")
    parser.add_argument("--pre_epochs", type=int, default=300, help="")
    parser.add_argument('--num_training', type=int, default=300, help='number of training circuits per section')
    parser.add_argument('--lr_for_predictor', type=float, default=0.001, help='learning rate')
    parser.add_argument('--bs', type=int, default=30, help='batch size for training')
    parser.add_argument("--numCir", type=int, default=300, help="number of labeled circuits")
    parser.add_argument("--numCir_unlabel", type=int, default=3000, help="number of unlabeled circuits")
    parser.add_argument("--ema_decay", type=float, default=0.99, help="EMA decay")
    parser.add_argument("--loss_weight", type=float, default=0.5, help="consistency")
    parser.add_argument("--infer_N", type=int, default=10, help="number of inference times for confidence")
    parser.add_argument("--top_k", type=int, default=300, help="top k samples")
    parser.add_argument('--consistency', type=float, default=1, metavar='WEIGHT', help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-rampup', type=int, default=5, metavar='EPOCHS', help='length of the consistency loss ramp-up')

    args = parser.parse_args()  # Parse command line arguments
    main(args)
    end_time = time.time()
