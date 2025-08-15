import numpy as np
import random
import tensorflow as tf
import tensorcircuit as tc
import matplotlib.pyplot as plt
import utils
from dataset_builder import shuffle
import os
import sys
import time
import config
from multiprocessing import Pool
import argparse


def circuit_eval(state, label, q_param, w_param, b_param, circuit, n_class, n_qubit):
    '''
    :param state: input state, shape (2**n_qb,)
    :param label: label, shape ()
    :param q_param: circuit's parameters
    :param w_param, b_param: postprocessing linear layer's parameters
    :param circuit: list, e.g., [['rz', 0], ['ry', 2], ['cz', 1, 2]]
    :param n_class: how many classes
    :param n_qubit: number of qubits
    :return: loss: shape ()
             ypred: prediction, shape () or (n_class,)
    '''
    #print('compiling................')
    cir = tc.Circuit(n_qubit, inputs=state)
    for idx, gate in enumerate(circuit):
        if gate[0] == 'cx':
            cir.cx(gate[1], gate[2])
        elif gate[0] == 'cz':
            cir.cz(gate[1], gate[2])
        elif gate[0] == 'rx':
            cir.rx(gate[1], theta=q_param[idx])
        elif gate[0] == 'ry':
            cir.ry(gate[1], theta=q_param[idx])
        elif gate[0] == 'rz':
            cir.rz(gate[1], theta=q_param[idx])
        elif gate[0] == 'x':
            cir.x(gate[1])
        elif gate[0] == 'sx':
            cir.unitary(gate[1], unitary=np.array([[0.5+0.5j, 0.5-0.5j], [0.5-0.5j, 0.5+0.5j]]), name="sx")
        else:
            print('undefined gate name found in function circuit_eval.')
            exit(123)

    outcome = []
    for i in range(n_qubit):
        exp_z = cir.expectation((tc.gates.z(), (i,)))
        exp_z = (tc.backend.real(exp_z) + 1) / 2.0
        outcome.append(exp_z)
    outcome = tc.backend.stack(outcome)    # shape (n_qubit,)
    ypred = tc.backend.reshape(outcome, [-1, 1])  # shape (n_qubit, 1)

    # apply a linear layer
    ypred = tc.backend.matmul(w_param, ypred)  # (1, 1) or (n_class, 1)
    ypred = tf.add(ypred, b_param)    # (1,1) or (n_class, 1)

    # compute loss
    if n_class == 2:
        ypred = tf.keras.activations.sigmoid(ypred)  # (1, 1)
        label = tc.backend.reshape(label, [-1, 1])  # (1, 1)
        criterion = tf.keras.losses.BinaryCrossentropy()
        loss = criterion(label, ypred)    # ()
        ypred = tf.squeeze(ypred)    # ()
    else:
        ypred = tc.backend.reshape(ypred, (1, -1))    # (1, n_class)
        ypred = tc.backend.softmax(ypred)    # (1, n_class)
        label_one_hot = tf.one_hot(label, depth=n_class)    # (n_class,)
        label_one_hot = tc.backend.reshape(label_one_hot, (1, -1))    # (1, n_class)
        loss = tf.keras.losses.categorical_crossentropy(label_one_hot, ypred)    # (1,)
        loss = tf.squeeze(loss)    # ()
        ypred = tf.squeeze(ypred)    # (n_class,)

    return loss, ypred

def get_qml_vvag():
    qml_vvag_train = tc.backend.vectorized_value_and_grad(circuit_eval, argnums=(2, 3, 4), vectorized_argnums=(0, 1), has_aux=True)
    qml_vvag_test = tc.backend.vmap(circuit_eval, vectorized_argnums=(0, 1))
    # don't jit when debugging, very slow
    qml_vvag_train = tc.backend.jit(qml_vvag_train, static_argnums=(5, 6, 7))
    qml_vvag_test = tc.backend.jit(qml_vvag_test, static_argnums=(5, 6, 7))
    return qml_vvag_train, qml_vvag_test

def train_one_circuit(work_queue):
    '''
    train one circuit for classification problems
    :param sample: [cir_list, depth, num_gate, num_2_qb_gate]
                    cir_list is a list, e.g., [['rz',0],['ry',1],['cz',1,2]]
    :param opt_cfg: a circuit's optimization configuration
    :param circuit_id: which circuit
    :param train_id: a circuit is trained for several times
    :param n_class: how many classes
    :param n_qubit: how many qubits
    :return: the test accuracy
    '''

    # circuit's parameters
    sample, opt_cfg, max_epoch, exp_dir, X_train, Y_train, X_test, \
        Y_test, circuit_id, n_class, n_qubit, run_id = work_queue[0], work_queue[1], work_queue[2], work_queue[3], work_queue[4], \
                                                       work_queue[5], work_queue[6], work_queue[7], work_queue[8], work_queue[9], work_queue[10], work_queue[11]
    np.random.seed(run_id)
    random.seed(run_id)
    tf.random.set_seed(run_id)
    tc.set_backend("tensorflow")
    tc.set_dtype("complex128")
    par = np.random.normal(loc=0, scale=1 / (4 * sample[1]) ** 0.5, size=sample[2])
    q_param = tf.Variable(initial_value=tf.convert_to_tensor(par, dtype=getattr(tf, tc.rdtypestr)))
    # postprocessing linear layer's parameters
    if n_class == 2:
        w_param = tf.Variable(initial_value=tf.random.normal(shape=(1, n_qubit), stddev=0.1, dtype=tf.float64))
        b_param = tf.Variable(initial_value=tf.random.normal(shape=(1, 1), stddev=0.1, dtype=tf.float64))
    else:
        w_param = tf.Variable(initial_value=tf.random.normal(shape=(n_class, n_qubit), stddev=0.1, dtype=tf.float64))
        b_param = tf.Variable(initial_value=tf.random.normal(shape=(n_class, 1), stddev=0.1, dtype=tf.float64))

    # load the optimization configuration
    opt_type = opt_cfg[0]
    lr = opt_cfg[1]
    batch_size = opt_cfg[2]

    if opt_type == 'Adagrad':
        opt = tf.keras.optimizers.Adagrad(lr)
    elif opt_type == 'Adam':
        opt = tf.keras.optimizers.Adam(lr)
    elif opt_type == 'Nadam':
        opt = tf.keras.optimizers.Nadam(lr)
    elif opt_type == 'RMSprop':
        opt = tf.keras.optimizers.RMSprop(lr)
    elif opt_type == 'SGD':
        opt = tf.keras.optimizers.SGD(lr)

    opt.build([q_param, w_param, b_param])

    qml_vvag_train, qml_vvag_test = get_qml_vvag()

    loss_list, test_acc_list = [], []
    loss_last = 1e10
    for epoch in range(1, max_epoch + 1):
        # update model
        running_loss = []
        order = np.random.permutation(len(X_train)).tolist()
        for batch in range(len(X_train) // batch_size):
            inputs = tf.gather(X_train, order[batch * batch_size: (batch + 1) * batch_size])    # shape (bs, 2**n_qubit)
            labels = tf.gather(Y_train, order[batch * batch_size: (batch + 1) * batch_size])    # shape (bs,)

            (loss, _), grad = qml_vvag_train(inputs, labels, q_param, w_param, b_param, sample[0], n_class, n_qubit)    # loss, shape (bs,)
            if grad[0] is not None:
                opt.apply_gradients([(grad[0]*2, q_param)])
            opt.apply_gradients([(grad[1], w_param)])
            opt.apply_gradients([(grad[2], b_param)])
            running_loss.extend(loss.numpy().tolist())    # record a batch's loss
        loss_list.append(sum(running_loss) / len(running_loss))    # record a epoch's average loss

        # compute test accuracy
        _, ypreds = qml_vvag_test(X_test, Y_test, q_param, w_param, b_param, sample[0], n_class, n_qubit)  # ypreds, shape (bs,) or (bs, n_class)
        ypreds = ypreds.numpy()  # (bs,) or (bs, n_class)
        if n_class == 2:
            ypreds = np.round(ypreds)    # (bs,)
        else:
            ypreds = np.argmax(ypreds, axis=1)  # (bs,)
        correct_predictions = np.sum(ypreds == Y_test.numpy())
        accuracy = correct_predictions / len(Y_test)
        test_acc_list.append(accuracy)  # record the average accuracy after an epoch's training

        # converge?
        if epoch % 10 == 0:
            distance = abs(loss_last - loss_list[-1])
            if distance < 0.0001:
                break
            else:
                loss_last = loss_list[-1]

    draw(loss_list, test_acc_list, exp_dir, circuit_id)

    return test_acc_list[-1]


def draw(loss_list, test_loss_list, exp_dir, circuit_id):
    '''
    :param loss_list: list, the train loss of a single training
    :param test_loss_list: list, the test loss of a single training
    :param exp_dir: the directory to save data, e.g., 'result/run_1/'
    :param circuit_id: which circuit
    :return:
    '''

    epochs = range(1, len(loss_list) + 1)

    plt.figure(figsize=(10, 6))

    # 绘制训练损失
    plt.subplot(2, 1, 1)  # 2行1列的第1个图
    plt.plot(epochs, loss_list, label='Training Loss', marker='o', markersize=1, color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # 绘制验证损失
    plt.subplot(2, 1, 2)  # 2行1列的第2个图
    plt.plot(epochs, test_loss_list, label='Test Loss', marker='o', markersize=1, color='orange')
    plt.title('Test Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    # 调整布局并显示图形
    plt.tight_layout()
    save_path_img = f'{exp_dir}/circuit_train_curve/'
    if not os.path.exists(save_path_img):
        os.makedirs(save_path_img)
    #plt.show()
    plt.savefig(save_path_img + f'circuit_{circuit_id}.png')
    plt.close()


def batch_train_parallel(device_name, task_name, n_class, max_epoch, run_id):
    start_time = time.time()

    # load task dataset
    dataset = utils.load_pkl(f'task_data/{task_name}.pkl')
    X_train = dataset['x_train']
    Y_train = dataset['y_train']
    X_test = dataset['x_test']
    Y_test = dataset['y_test']

    X_train = tf.convert_to_tensor(X_train, dtype=tf.complex128)
    Y_train = tf.convert_to_tensor(Y_train, dtype=tf.int32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.complex128)
    Y_test = tf.convert_to_tensor(Y_test, dtype=tf.int32)

    # load finetune circuits
    exp_dir = f'result/cir_sample/{device_name}/finetune/run_{run_id}/'
    samples = utils.load_pkl(f'{exp_dir}samples.pkl')
    opt_cfgs = utils.load_pkl(f'{exp_dir}opt_cfg.pkl')

    # load labels if exists
    exp_dir = f'result/{task_name}/{device_name}/run_{run_id}/'
    if os.path.exists(f'{exp_dir}labels.pkl'):
        labels = utils.load_pkl(f'{exp_dir}labels.pkl')
        ft_cir_range = range(len(labels), len(samples))
    else:
        labels = []
        ft_cir_range = range(len(samples))

    # start training
    work_queue = []
    for i in ft_cir_range:
        print(f'circuit id: {i}')
        # load finetune circuits
        exp_dir = f'result/{task_name}/{device_name}/run_{run_id}/'
        work_queue.append([samples[i], opt_cfgs[i], max_epoch, exp_dir,
                                     X_train, Y_train, X_test, Y_test, i, n_class,
                                     config.device[device_name]['n_qubit'], run_id])

    pool = Pool(processes=10)
    result = pool.map(train_one_circuit, work_queue)
    pool.close()
    pool.join()
    utils.save_pkl(result, f'{exp_dir}labels_add_1.pkl')
    labels.extend(result)
    utils.save_pkl(labels, f'{exp_dir}labels.pkl')

    end_time = time.time()
    duration = end_time - start_time
    print(f"run time: {int(duration // 3600)} hours {int((duration % 3600) // 60)} minutes")


def batch_parallel(samples, opt_cfgs, device_name, task_name, max_epoch, n_class, run_ids, mask_ratio):
    start_time = time.time()

    # load dataset
    dataset = utils.load_pkl(f'task_data/{task_name}.pkl')
    X_train = dataset['x_train']
    Y_train = dataset['y_train']
    X_test = dataset['x_test']
    Y_test = dataset['y_test']

    X_train = tf.convert_to_tensor(X_train, dtype=tf.complex128)
    Y_train = tf.convert_to_tensor(Y_train, dtype=tf.int32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.complex128)
    Y_test = tf.convert_to_tensor(Y_test, dtype=tf.int32)

    # start training
    labels = []
    work_queue = []

    for i in range(len(samples)):
        print(f'circuit id: {i}')
        # load finetune circuits
        exp_dir = f'result/{task_name}/{device_name}/run_{run_ids[i]}/mask_{int(100*mask_ratio)}/'
        work_queue.append([samples[i], opt_cfgs[i], max_epoch, exp_dir,
                                     X_train, Y_train, X_test, Y_test, i, n_class,
                                     config.device[device_name]['n_qubit'], run_ids[i]])

    # result = train_one_circuit(work_queue[0][0], work_queue[0][1], work_queue[0][2], work_queue[0][3], work_queue[0][4],
    #                              work_queue[0][5], work_queue[0][6], work_queue[0][7], work_queue[0][8], work_queue[0][9],
    #                              work_queue[0][10], work_queue[0][11])
    pool = Pool(processes=10)
    result = pool.map(train_one_circuit, work_queue)
    pool.close()
    pool.join()

    # utils.save_pkl(test_acc, f'{exp_dir}labels.pkl')

    end_time = time.time()
    duration = end_time - start_time
    print(f"run time: {int(duration // 3600)} hours {int((duration % 3600) // 60)} minutes")
    return result



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument('--device_name', type=str, default='grid_8q', help='')
    parser.add_argument('--num_layer', type=int, default=10, help='')
    args = parser.parse_args()

    device_name = 'ishape'    # see config.py
    task_name = 'FMNIST_7q'
    n_class = 10
    max_epoch = 50

    run_id = 1
    batch_train_parallel(device_name, task_name, n_class, max_epoch, run_id)
    run_id = 2
    batch_train_parallel(device_name, task_name, n_class, max_epoch, run_id)
    run_id = 3
    batch_train_parallel(device_name, task_name, n_class, max_epoch, run_id)
    run_id = 4
    batch_train_parallel(device_name, task_name, n_class, max_epoch, run_id)
    run_id = 5
    batch_train_parallel(device_name, task_name, n_class, max_epoch, run_id)


    device_name = 'grid_8q'  # see config.py
    task_name = 'FMNIST_8q'

    run_id = 1
    batch_train_parallel(device_name, task_name, n_class, max_epoch, run_id)
    run_id = 2
    batch_train_parallel(device_name, task_name, n_class, max_epoch, run_id)
    run_id = 3
    batch_train_parallel(device_name, task_name, n_class, max_epoch, run_id)
    run_id = 4
    batch_train_parallel(device_name, task_name, n_class, max_epoch, run_id)
    run_id = 5
    batch_train_parallel(device_name, task_name, n_class, max_epoch, run_id)







