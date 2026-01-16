import pandas as pd
from pathlib import Path
from datetime import datetime
from simulator.Simulator import Simulator

if __name__ == '__main__':

    total_experiments = 0

    datasets        = ['EMNIST']
    algorithms      = ['ifca'] #['fedavg','fedprox', 'scaffold', 'ifca']
    areas           = [3, 5, 9]
    partitionings   = ['Hard', 'Dirichlet']
    sparsifications = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    clients         = 50
    batch_size      = 32
    local_epochs    = 2
    global_rounds   = 60
    data_dir        = 'data'
    max_seed        = 5

    data_output_directory = Path(data_dir)
    data_output_directory.mkdir(parents=True, exist_ok=True)

    csv_file = f'finished_experiment_log.csv'
    df = pd.DataFrame(columns=['timestamp', 'experiment'])
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        pass

    for seed in range(max_seed):
        for partitioning in partitionings:
            for algorithm in algorithms:
                for dataset in datasets:
                    for area in areas:
                        for sparsification in sparsifications:
                            simulator = Simulator(algorithm, partitioning, area, dataset, clients, batch_size, local_epochs, data_dir, seed, sparsification)
                            simulator.seed_everything(seed)
                            simulator.start(global_rounds)
                            experiment_name = f'seed-{seed}_regions-{area}_algorithm-{algorithm}_sparsity-{sparsification}_partitioning-{partitioning}_dataset-{dataset}'
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            new_line = {'timestamp': timestamp, 'experiment': experiment_name}
                            df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)
                            df.to_csv(csv_file, index=False)