from common_utils import *
from params import configs
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from tqdm import tqdm
from data_utils import pack_data_from_config
import time
import numpy as np
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id


def test_heuristic_method(data_set, heuristic, seed):
    """
        test one heuristic method on the given data
    :param data_set:  test data
    :param heuristic: the name of heuristic method
    :param seed: seed for testing
    :return: the test results including the makespan and time
    """
    setup_seed(seed)
    result = []

    for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
        n_j = data_set[0][i].shape[0]
        n_op, n_m = data_set[1][i].shape
        env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)

        env.set_initial_data([data_set[0][i]], [data_set[1][i]])

        t1 = time.time()
        while True:
            action = heuristic_select_action(heuristic, env)

            _, _, done = env.step(np.array([action]))

            if done:
                break

        t2 = time.time()
        # tqdm.write(f'Instance {i + 1} , makespan:{-ep_reward} , time:{t2 - t1}')
        result.append([env.current_makespan[0], t2 - t1])

    return np.array(result)


def main():
    """
        test heuristic methods following the config and save the results:
        here are heuristic methods selected for comparison:

        FIFO: First in first out
        MOR(or MOPNR): Most operations remaining
        SPT: Shortest processing time
        MWKR: Most work remaining
    """
    setup_seed(configs.seed_test)
    if not os.path.exists('./test_results'):
        os.makedirs('./test_results')

    test_data = pack_data_from_config(configs.data_source, configs.test_data)

    if len(configs.test_method) == 0:
        test_method = ['FIFO', 'MOR', 'SPT', 'MWKR']
    else:
        test_method = configs.test_method

    for data in test_data:
        print("-" * 25 + "Test Heuristic Methods" + "-" * 25)
        print('Test Methods:', test_method)
        print(f"test data name: {configs.data_source},{data[1]}")
        save_direc = f'./test_results/{configs.data_source}/{data[1]}'

        if not os.path.exists(save_direc):
            os.makedirs(save_direc)
        for method in test_method:
            save_path = save_direc + f'/Result_{method}_{data[1]}.npy'

            if (not os.path.exists(save_path)) or configs.cover_heu_flag:
                print(f"Heuristic method : {method}")
                seed = configs.seed_test

                result_5_times = []
                # test 5 times, record average makespan and time.
                for j in range(5):
                    result = test_heuristic_method(data[0], method, seed + j)
                    result_5_times.append(result)

                    print(f"the {j + 1}th makespan:", np.mean(result[:, 0]))
                result_5_times = np.array(result_5_times)
                save_result = np.mean(result_5_times, axis=0)
                print(f"testing results of {method}:")
                print(f"makespan(sampling): ", save_result[:, 0].mean())
                print(f"time: ", save_result[:, 1].mean())
                np.save(save_path, save_result)


if __name__ == '__main__':
    main()
