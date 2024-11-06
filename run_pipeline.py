
import subprocess
import argparse


def run_command(command):
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, check=True)

def main(args):
    initial_seed, dataset, gpu_id = args.seed, args.dataset, args.gpu_id
    use_augmentation, augmentation_ratio, label_aug_proportions = args.use_augmentation, args.augmentation_ratio, args.label_aug_proportions
    unlabel_ratio, label_train_ratio, label_val_ratio, label_test_ratio = args.unlabel_ratio, args.label_train_ratio, args.label_val_ratio, args.label_test_ratio
    

    for i in range(5):
        seed = initial_seed + i
        print(f"\nStarting iteration {i + 1} with seed={seed}\n")

        commands = [
            f"python download_and_split_tudataset.py --seed={seed} --dataset={dataset} --unlabel_ratio={unlabel_ratio} --label_train_ratio={label_train_ratio} --label_val_ratio={label_val_ratio} --label_test_ratio={label_test_ratio}",
            f"CUDA_VISIBLE_DEVICES={gpu_id} python train_gnn_encoder.py --seed={seed} --dataset={dataset}",
            f"CUDA_VISIBLE_DEVICES={gpu_id} python main.py --training_mode=supervised --seed={seed} --dataset={dataset}",
            f"CUDA_VISIBLE_DEVICES={gpu_id} python infer_and_prepare_semi_supervised_data.py --seed={seed} --dataset={dataset}",
            f"CUDA_VISIBLE_DEVICES={gpu_id} python main.py --training_mode=semi_supervised --seed={seed} --dataset={dataset} --use_augmentation={use_augmentation} --augmentation_ratio={augmentation_ratio} --label_aug_proportions={label_aug_proportions}"
        ]

        for command in commands:
            try:
                run_command(command)
            except subprocess.CalledProcessError:
                print("Execution stopped due to error.")
                return 


    print("\nExecuting statistics script...\n")
    try:
        run_command(f"python calculate_statistics.py --dataset={dataset}")
    except subprocess.CalledProcessError:
        print("Statistics script execution stopped due to error.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run GNN Pipeline 5 times with increasing seed values")
    parser.add_argument("--seed", type=int, required=True, help="The initial seed value for the experiments")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset name for the experiments")
    parser.add_argument("--use_augmentation", type=int, default=1, help="Use data augmentation")
    parser.add_argument("--augmentation_ratio", type=float, default=0.5)
    parser.add_argument('--label_aug_proportions', type=str, default=None, help='Label augmentation proportions in the format label1:prop1,label2:prop2')
    parser.add_argument('--unlabel_ratio', type=float, default=0.5, help='Ratio of unlabeled data')
    parser.add_argument('--label_train_ratio', type=float, default=0.2, help='Ratio of labeled training data')
    parser.add_argument('--label_val_ratio', type=float, default=0.15, help='Ratio of labeled validation data')
    parser.add_argument('--label_test_ratio', type=float, default=0.15, help='Ratio of labeled test data')
    parser.add_argument("--gpu_id", type=int, default=0, help="The CUDA device ID to use (e.g., '0')")

    args = parser.parse_args()

    main(args)


