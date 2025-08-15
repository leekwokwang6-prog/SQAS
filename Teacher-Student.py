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

tc.set_dtype("complex128")
tc.set_backend("tensorflow")


def normalize_features(x):
    '''
    Normalize each sample (row), by subtracting mean and dividing by std of its features.
    :param x: numpy array, shape (batch_size, -1)
    :return: numpy array, shape (batch_size, -1)
    '''
    mean = np.mean(x, axis=0, keepdims=True)  # Calculate the mean of each row
    std = np.std(x, axis=0, keepdims=True) + 1e-8  # Calculate the standard deviation of each row and add a small value to avoid division by zero
    return (x - mean) / std


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.loss_weight * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def confident_choose(arg, properties_train_unlabeled, GP_predictor, properties_train, y_train):
    """
    Perform infer_N Monte Carlo inferences for all unlabeled data,
    calculate the mean and standard deviation of the output for each sample,
    and then take the top_k with the smallest standard deviation as pseudo-labeled samples.
    Return a new training DataLoader with these top_k pseudo-labeled samples added.
    """
    # Ensure the model is in train mode to enable Dropout
    GP_predictor.train()
    with torch.no_grad():
        all_outputs = []
        for _ in range(arg.infer_N):
            out = GP_predictor(properties_train_unlabeled)  # [N_unlabeled]
            all_outputs.append(out.detach().cpu())
        all_outputs = torch.stack(all_outputs, dim=0)

        # Calculate the standard deviation and mean for each sample
        stds = all_outputs.std(dim=0)  # [N_unlabeled]
        means = all_outputs.mean(dim=0)  # [N_unlabeled]

        # Select top_k samples with the smallest standard deviation
        _, topk_idxs = torch.topk(-stds, arg.top_k)  # Negative sign to get smallest values
        selected_props = properties_train_unlabeled[topk_idxs].cpu()  # [top_k, ...]
        selected_pseudos = means[topk_idxs]  # [top_k]

        # Append to the original labeled dataset
        properties_train_new = torch.cat([properties_train.cpu(), selected_props], dim=0)
        y_train_new = torch.cat([y_train.cpu(), selected_pseudos], dim=0)

        # Build the new training set
        train_data = TensorDataset(properties_train_new, y_train_new)
        return train_data


def PQAS_GP_search_test(data_cir, arg):
    properties_train, y_train = data_cir['train']['properties_list'], data_cir['train']['energy']
    properties_train_unlabeled = data_cir['test']['properties_list']  # Unlabeled samples
    properties_eval_unlabeled = data_cir['test_eval']['properties_list']  # Unlabeled samples
    properties_test, y_test = data_cir['val']['properties_list'], data_cir['val']['energy']

    if arg.properties_normal:
        properties_train = normalize_features(np.array(properties_train))  # Normalize training set
        properties_train_unlabeled = normalize_features(np.array(properties_train_unlabeled))
        properties_eval_unlabeled = normalize_features(np.array(properties_eval_unlabeled))
        properties_test = normalize_features(np.array(properties_test))  # Normalize test set

    # Convert to Tensor
    properties_train = torch.Tensor(properties_train)
    properties_train_unlabeled = torch.Tensor(properties_train_unlabeled)
    properties_eval_unlabeled = torch.Tensor(properties_eval_unlabeled)
    properties_test = torch.Tensor(properties_test)

    y_train = normalize_features(np.array(y_train))
    y_test = normalize_features(np.array(y_test))
    y_train = torch.Tensor(y_train)
    y_test = torch.Tensor(y_test)

    # Create training data loaders
    train_data_warm = TensorDataset(properties_train, y_train)
    train_warmup_loader = DataLoader(train_data_warm, batch_size=arg.bs, shuffle=True, drop_last=False)

    train_data_labeled = TensorDataset(properties_train, y_train)
    train_labeled_loader = DataLoader(train_data_labeled, batch_size=arg.bs, shuffle=True, drop_last=False)

    train_data_unlabeled = TensorDataset(properties_train_unlabeled)
    train_unlabeled_loader = DataLoader(train_data_unlabeled, batch_size=arg.bs, shuffle=True, drop_last=False)

    # Initialize predictor
    GP_predictor = MLP(arg.loop, 10, arg.hidden, 1, arg.dropout)  # Assume MLP is predefined model
    GP_predictor_TS = MLP(arg.loop, 10, arg.hidden, 1, arg.dropout)  # Assume MLP is predefined model

    optimizer = torch.optim.Adam(GP_predictor.parameters(), lr=arg.lr_for_predictor)
    loss_func_mse = torch.nn.MSELoss()
    loss_func = torch.nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    total_loss_train = []
    loss_EVAL = []
    loss_TRAIN = []
    loss_std_train_sup = []
    loss_std_eval_sup = []
    loss_ema_train_sup = []
    loss_ema_eval_sup = []
    spearman_coeffs = []
    spearman_coeffs_evl = []
    ema_time = []
    global_step = 0

    # 1. Warmup training process
    for epoch in range(arg.warmup_epochs):
        GP_predictor.train()
        train_loss = []
        for step, (b_properties, b_y) in enumerate(train_warmup_loader):
            optimizer.zero_grad()
            b_properties, b_y = b_properties, b_y
            # Forward propagation
            forward = GP_predictor(b_properties)  # Output of main model
            loss = loss_func(forward, b_y)  # Supervised loss
            # Backpropagation
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

            loss_std_train_sup.append(loss_train)
            loss_std_eval_sup.append(loss_eval)
            loss_ema_train_sup.append(loss_train)
            loss_ema_eval_sup.append(loss_eval)

            coeff_train, _ = spearmanr(forward_train.detach().numpy(), y_train.detach().numpy())
            coeff_evl, _ = spearmanr(forward.detach().numpy(), y_test.detach().numpy())

            spearman_coeffs.append(coeff_train)
            spearman_coeffs_evl.append(coeff_evl)

            total_loss_train.append(np.mean(train_loss))

    start_time_ema = time.time()
    GP_predictor_TS.load_state_dict(GP_predictor.state_dict())  # Copy student model parameters to teacher model

    # Get an iterator
    unlabeled_iter = iter(train_unlabeled_loader)

    # 2. Main training process
    optimizer_TS = torch.optim.Adam(GP_predictor.parameters(), lr=arg.lr_for_predictor)
    scheduler_TS = ReduceLROnPlateau(optimizer_TS, 'min', factor=0.1, patience=10, verbose=True)

    con = []
    for epoch in range(arg.pre_epochs):
        temp = []
        GP_predictor.train()
        GP_predictor_TS.train()
        train_loss = []

        for step, (b_properties, b_y) in enumerate(train_labeled_loader):
            optimizer_TS.zero_grad()

            # Fetch one batch from unlabeled_iter; reset if exhausted
            b_properties, b_y = b_properties, b_y  # 64 labeled samples
            try:
                b_properties_unlabeled = next(unlabeled_iter)[0]
            except StopIteration:
                unlabeled_iter = iter(train_unlabeled_loader)
                b_properties_unlabeled = next(unlabeled_iter)[0]

            # Forward propagation
            forward = GP_predictor(b_properties)
            b_properties_unlabeled = torch.cat([b_properties_unlabeled, b_properties], dim=0)
            forward_unlabeled = GP_predictor(b_properties_unlabeled)
            forward_unlabeled_ema = GP_predictor_TS(b_properties_unlabeled)

            # Supervised loss
            loss_sup = loss_func(forward, b_y)
            loss_consistency = loss_func_mse(forward_unlabeled, forward_unlabeled_ema)
            temp.append(loss_consistency.detach().numpy())

            consistency_weight = get_current_consistency_weight(epoch)
            loss = loss_sup + consistency_weight * loss_consistency

            # Backpropagation
            loss.backward()
            optimizer_TS.step()

            train_loss.append(loss.item())
            global_step += 1
            update_ema_variables(GP_predictor, GP_predictor_TS, arg.ema_decay, global_step)

        scheduler_TS.step(np.average(train_loss))
        con.append(np.average(temp))

        GP_predictor.eval()
        GP_predictor_TS.eval()
        with torch.no_grad():
            # Validation set
            forward_train = GP_predictor(properties_train)
            forward_train_unlabel = GP_predictor(properties_train_unlabeled)
            forward_train_unlabel_ema = GP_predictor_TS(properties_train_unlabeled)
            loss_sup_train = loss_func(forward_train, y_train)
            loss_consistency_train = loss_func_mse(forward_train_unlabel, forward_train_unlabel_ema)
            loss_train = loss_sup_train + arg.loss_weight * loss_consistency_train
            loss_TRAIN.append(loss_train)

            forward_eval = GP_predictor(properties_test)
            forward_eval_unlabel = GP_predictor(properties_eval_unlabeled)
            forward_eval_unlabel_ema = GP_predictor_TS(properties_eval_unlabeled)
            loss_sup_eval = loss_func(forward_eval, y_test)
            loss_consistency_eval = loss_func_mse(forward_eval_unlabel, forward_eval_unlabel_ema)
            loss_eval = loss_sup_eval + arg.loss_weight * loss_consistency_eval
            loss_EVAL.append(loss_eval)

            # Supervised loss parts
            loss_std_train_sup.append(loss_sup_train)
            loss_std_eval_sup.append(loss_sup_eval)

            # EMA supervised loss
            forward_train_ema = GP_predictor_TS(properties_train)
            loss_sup_train_ema = loss_func(forward_train_ema, y_train)
            loss_ema_train_sup.append(loss_sup_train_ema)
            forward_eval_ema = GP_predictor_TS(properties_test)
            loss_sup_eval_ema = loss_func(forward_eval_ema, y_test)
            loss_ema_eval_sup.append(loss_sup_eval_ema)

            coeff_train, _ = spearmanr(forward_train_ema.detach().numpy(), y_train.detach().numpy())
            coeff_evl, _ = spearmanr(forward_eval_ema.detach().numpy(), y_test.detach().numpy())
            spearman_coeffs.append(coeff_train)
            spearman_coeffs_evl.append(coeff_evl)

            total_loss_train.append(np.mean(train_loss))

    GP_predictor_TS.eval()
    with torch.no_grad():
        pred_test = GP_predictor_TS(properties_test).cpu().numpy()
        pred_train = GP_predictor_TS(properties_train).cpu().numpy()

    end_time_ema = time.time()
    ema_time.append(end_time_ema - start_time_ema)

    coeff1, _ = spearmanr(GP_predictor_TS(properties_train).detach().numpy(), y_train.detach().numpy())
    print('Training set SP correlation coefficient:', coeff1)
    coeff_evl1, _ = spearmanr(GP_predictor_TS(properties_test).detach().numpy(), y_test.detach().numpy())
    print('Validation set SP correlation coefficient:', coeff_evl1)

    metrics_path_file = f'saved_models_Teacher-Student_{arg.device}_{arg.task}'
    if not os.path.exists(metrics_path_file):
        os.makedirs(metrics_path_file)

    GP_predictor_TS.eval()
    # Here we can add: save the state_dict of GP_predictor
    save_path = f"{metrics_path_file}/predictor_Teacher-Student_seed{arg.seed}.pth"
    torch.save(GP_predictor.state_dict(), save_path)
    print(f" >>> Parameters saved to: {save_path}")

    # Compute ranking
    rank = np.argsort(np.array(pred_test))
    return rank, pred_test, pred_train, coeff_evl1


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

    # Load data (validation set)
    data_cir['val']['architecture'] = utils.load_pkl(save_datasets + f'training/run_{(arg.seed + 1) % 5}/samples.pkl')[:arg.numCir]
    data_cir['val']['matrix_cir'] = utils.load_pkl(save_datasets + f'training/run_{(arg.seed + 1) % 5}/adj_feat_matrix.pkl')[:arg.numCir]
    data_cir['val']['energy'] = utils.load_pkl(save_datasets + f'training/run_{(arg.seed + 1) % 5}/energy.pkl')[:arg.numCir]
    data_cir['val']['properties_list'] = utils.load_pkl(save_datasets + f'training/run_{(arg.seed + 1) % 5}/properties_list.pkl')[:arg.numCir]

    # Load data (test set)
    data_cir['test']['architecture'] = utils.load_pkl(save_datasets + f'search_pool/run_{arg.seed}/samples.pkl')[:arg.numCir_unlabel]
    data_cir['test']['matrix_cir'] = utils.load_pkl(save_datasets + f'search_pool/run_{arg.seed}/adj_feat_matrix.pkl')[:arg.numCir_unlabel]
    data_cir['test']['properties_list'] = utils.load_pkl(save_datasets + f'search_pool/run_{arg.seed}/properties_list.pkl')[:arg.numCir_unlabel]

    # Load data (test_eval set)
    data_cir['test_eval']['architecture'] = utils.load_pkl(save_datasets + f'search_pool/run_{(arg.seed + 1) % 5}/samples.pkl')[:arg.numCir_unlabel]
    data_cir['test_eval']['matrix_cir'] = utils.load_pkl(save_datasets + f'search_pool/run_{(arg.seed + 1) % 5}/adj_feat_matrix.pkl')[:arg.numCir_unlabel]
    data_cir['test_eval']['properties_list'] = utils.load_pkl(save_datasets + f'search_pool/run_{(arg.seed + 1) % 5}/properties_list.pkl')[:arg.numCir_unlabel]

    rank, pred, pred_train, coeff_evl1 = PQAS_GP_search_test(data_cir, arg)

    """Performance evaluation"""
    return coeff_evl1


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument('--task', type=str, default='TFIM_8', help='see configs for available tasks')
    parser.add_argument('--device', type=str, default='grid_16q', help='')
    parser.add_argument('--method', type=str, default='Teacher-Student', help='Search method')
    parser.add_argument("--dropout", type=float, default=0.3, help='dropout')
    parser.add_argument("--loop", type=int, default=2, help='')
    parser.add_argument("--hidden", type=int, default=64, help='')
    parser.add_argument('--properties_normal', type=int, default=1, help='Whether to load circuit file from specified location')
    parser.add_argument("--warmup_epochs", type=int, default=300, help="")
    parser.add_argument("--pre_epochs", type=int, default=200, help="")
    parser.add_argument('--lr_for_predictor', type=float, default=0.001, help='learning rate')
    parser.add_argument('--bs', type=int, default=30, help='batch size for training')
    parser.add_argument("--numCir", type=int, default=300, help="Number of labeled dataset")
    parser.add_argument("--numCir_unlabel", type=int, default=100000, help="Number of unlabeled dataset")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay")
    parser.add_argument("--loss_weight", type=float, default=0.3, help="consistency")
    parser.add_argument("--infer_N", type=int, default=10, help="Number of inference times for confidence estimation")
    parser.add_argument("--top_k", type=int, default=200, help="Top_K")
    parser.add_argument('--consistency', type=float, default=1, metavar='WEIGHT', help='use consistency loss with given weight (default: None)') #200
    parser.add_argument('--consistency-rampup', type=int, default=20, metavar='EPOCHS', help='length of the consistency loss ramp-up')

    args = parser.parse_args()
    main(args)
    end_time = time.time()
