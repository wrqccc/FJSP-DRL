from params import configs
import os
import numpy as np
import pandas as pd
import time as time

null_val = np.nan
str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
non_exist_data = []
winner = []


def load_result(source, model_name, data_name):
    """
        load the testing results.
    :param source: the source of data
    :param model_name: the name of model
    :param data_name: the name of data
    :return: makespan, average makespan, average time
    """
    file_path = f'./test_results/{source}/{data_name}/Result_{model_name}_{data_name}.npy'
    if os.path.exists(file_path):
        test_result = np.load(file_path)
        make_span = test_result[:, 0]
        make_span_mean = np.mean(make_span)
        time_mean = np.mean(test_result[:, 1])
        return make_span, make_span_mean, time_mean
    else:
        # ++ non_exist_data
        non_exist_data.append([source, f'Result_{model_name}_{data_name}.npy'])
        return [], null_val, null_val


def load_solution_by_or(source, data_name):
    """
        load the results solved by OR-Tools
    :param source: the source of data
    :param data_name: the name of data
    :return: makespan, average makespan, average time, optimal percentage
    """
    file_path = f'./or_solution/{source}/solution_{data_name}.npy'
    if os.path.exists(file_path):
        solution = np.load(file_path)
        or_make_span = solution[:, 0]
        or_make_span_mean = np.mean(or_make_span)
        or_time_mean = np.mean(solution[:, 1])
        # print(solution[:, 1])
        or_percentage = np.where(solution[:, 1] < configs.max_solve_time)[0].shape[0] / solution[:, 1].shape[0]
        return or_make_span, or_make_span_mean, or_time_mean, or_percentage
    else:
        # ++ non_exist_data
        non_exist_data.append([source, f'solution_{data_name}.npy'])
        return [], null_val, null_val, null_val


def load_benchmark_solution(data_name):
    """
        load the best solutions of benchmark data from files
    :param data_name: the name of benchmark data
    :return: makespan, average makespan
    """
    file_path = f'./data/bench_data/BenchDataSolution.csv'
    bench_data = pd.read_csv(file_path)
    make_span = bench_data.loc[bench_data['benchname'] == data_name, 'ub'].values
    make_span_mean = np.mean(make_span)
    return make_span, make_span_mean


def print_test_results_to_excel(source, data_list, model_list, file_name=f"test_{str_time}"):
    """
    :param file_name: the name of saved file
    :param source: the source of data
    :param data_list: the list of data name
    :param model_list: the list of model name
    """
    if not os.path.exists(f'./TestDataToExcel/{source}'):
        os.makedirs(f'./TestDataToExcel/{source}')

    idx = np.append('Ortools', model_list)
    columns = data_list

    make_span_form = []
    gap_form = []
    time_form = []

    optimal_make_span_list = []

    or_make_span_list = []
    or_time_list = []
    or_gap_list = []
    or_percentage_list = []

    # Get Optimal Solution
    for data_name in data_list:
        or_make_span, or_make_span_mean, or_time, or_percentage = load_solution_by_or(source, data_name)
        or_make_span_list.append(or_make_span_mean)
        or_time_list.append(or_time)
        or_percentage_list.append(or_percentage)
        if source == 'BenchData':
            bench_make_span, bench_make_span_mean = load_benchmark_solution(data_name)
            optimal_make_span_list.append(bench_make_span)
            if len(or_make_span) == 0:
                gap = null_val
            else:
                # gap = np.maximum(or_make_span / bench_make_span_mean, bench_make_span_mean / or_make_span)
                # gap = '{:.2f}%'.format((or_make_span / bench_make_span_mean - 1) * 100)
                gap = np.mean((or_make_span / bench_make_span - 1) * 100)
            or_gap_list.append(gap)
        else:
            optimal_make_span_list.append(or_make_span)

    # print
    for model_name in model_list:
        make_span_row = []
        gap_row = []
        time_row = []
        for i, data_name in enumerate(data_list):
            make_span, make_span_mean, time = load_result(source, model_name, data_name)
            # 判断gap是否存在
            if len(optimal_make_span_list[i]) == 0 or len(make_span) == 0:
                gap = null_val
            else:
                # gap = np.maximum(make_span_mean / optimal_make_span_list[i], optimal_make_span_list[i] / make_span_mean)
                # gap = '{:.2f}%'.format((make_span_mean / optimal_make_span_list[i] - 1) * 100)
                gap = np.mean((make_span / optimal_make_span_list[i] - 1) * 100)
            make_span_row.append(make_span_mean)
            gap_row.append(gap)
            time_row.append(time)

        make_span_form.append(make_span_row)
        gap_form.append(gap_row)
        time_form.append(time_row)

    # to excel
    writer = pd.ExcelWriter(f'./TestDataToExcel/{source}/{file_name}.xlsx')
    # insert or-tools result
    make_span_form.insert(0, or_make_span_list)
    time_form.insert(0, or_time_list)

    # console message
    print('=' * 25 + 'DataToExcelMessage' + '=' * 25)
    print(f'source:{source}')
    print(f'model_list:{np.array(model_list)}')
    print(f'data_list:{np.array(data_list)}')

    # Sheet1: make_span_mean

    make_span_file = pd.DataFrame(make_span_form, columns=columns, index=idx)

    # Sheet2: gap
    if source == 'BenchData':
        gap_form.insert(0, or_gap_list)
        gap_file = pd.DataFrame(gap_form, columns=columns, index=idx)

    else:
        gap_file = pd.DataFrame(gap_form, columns=columns, index=idx[1:])

    # Sheet3: time
    time_file = pd.DataFrame(time_form, columns=columns, index=idx)

    # Sheet4: optimal percentage
    or_percentage_file = pd.DataFrame([or_percentage_list], columns=columns, index=['percentage'])

    # Sheet5: the model that performs best on the test data
    winner_pd = pd.DataFrame(make_span_file.iloc[1:].idxmin(), columns=['winner'])

    # Sheet6: data that are non-existent
    non_exist_data_pd = pd.DataFrame(non_exist_data, columns=['source', 'filename'])

    # whether sort the results by makespan
    if len(data_list) == 1 and configs.sort_flag:
        make_span_file = make_span_file.sort_values(by=data_list[0])
        gap_file = gap_file.sort_values(by=data_list[0])
        time_file = time_file.sort_values(by=data_list[0])

    # format transform
    for i in range(gap_file.shape[0]):
        for j in range(gap_file.shape[1]):
            gap_file.iloc[i, j] = f"{round(gap_file.iloc[i, j], 2)}%"
    # data_save
    make_span_file.to_excel(writer, sheet_name='makespan', index=True)
    gap_file.to_excel(writer, sheet_name='gap', index=True)
    time_file.to_excel(writer, sheet_name='time', index=True)
    or_percentage_file.to_excel(writer, sheet_name='or_percentage', index=True)
    winner_pd.to_excel(writer, sheet_name='winner', index=True)
    non_exist_data_pd.to_excel(writer, sheet_name='nonExistData', index=True)

    writer.close()
    # winner & non-exist data to console
    print(make_span_file)
    print('\n Successfully print data to excel! \n')
    print('=' * 50)
    print('winner:\n')
    print(winner_pd)
    print('\nnon_exist_data:\n\n')
    print(non_exist_data_pd)


def main():
    """
        print the testing results (Obj./Gap/Time) to files (excel)
        data: {configs.test_data}
        source: {configs.data_source}
    """
    model_list = []
    for data_name in configs.test_data:
        result_list = os.listdir(f'./test_results/{configs.data_source}/{data_name}')
        model_list = model_list + [result.split('_')[1] for result in result_list]

    model_list = list(set(model_list))
    file_name = f"test_{str_time}_{configs.data_source}_{configs.test_data[0]}"
    print_test_results_to_excel(configs.data_source, configs.test_data, model_list, file_name)


if __name__ == '__main__':
    main()
