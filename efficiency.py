import time
import torch
import numpy as np
from memory_profiler import memory_usage
import inference_2D
import inference_3D
from os.path import join, isfile, basename
from glob import glob
import tracemalloc
tracemalloc.start()

parser = argparse.ArgumentParser()

parser.add_argument(
    '-inference_function',
    type=str,
    default="inference_3D",
    help='inference function to measure efficiency of',
    required=True
)

parser.add_argument(
    '-data_root',
    type=str,
    default=None,
    help='root directory of the data',
    required=True
)
parser.add_argument(
    '-pred_save_dir',
    type=str,
    default=None,
    help='directory to save the prediction',
    required=True
)
parser.add_argument(
    '-medsam_lite_checkpoint_path',
    type=str,
    default="lite_medsam.pth",
    help='path to the checkpoint of MedSAM-Lite',
    required=True
)
parser.add_argument(
    '-device',
    type=str,
    default="cpu",
    help='device to run the inference',
)
parser.add_argument(
    '-num_workers',
    type=int,
    default=4,
    help='number of workers for inference with multiprocessing',
)
parser.add_argument(
    '--save_overlay',
    action='store_true',
    help='whether to save the overlay image'
)
parser.add_argument(
    '-png_save_dir',
    type=str,
    default='./overlay',
    help='directory to save the overlay image'
)
parser.add_argument(
    '--overwrite',
    action='store_true',
    help='whether to overwrite the existing prediction'
)

args = parser.parse_args()

inference_function = args.inference_function
data_root = args.data_root
pred_save_dir = args.pred_save_dir
save_overlay = args.save_overlay
num_workers = args.num_workers
overwrite = args.overwrite
medsam_lite_checkpoint_path = args.medsam_lite_checkpoint_path

def measure_efficiency(inference_function, *args, **kwargs):
    # Time Profiling
    start_time = time.time()
    ##results = inference_function(*args, **kwargs)
    end_time = time.time()
    total_time = end_time - start_time

    # Memory Usage
    mem_usage = memory_usage((inference_function, args, kwargs))
    peak_memory = max(mem_usage)

    # Throughput
    gt_path_files = sorted(glob(join(data_root, '*.npz'), recursive=True))
    batch_size = len(gt_path_files)
    throughput = batch_size / total_time if total_time > 0 else 0

    # Print and save the results
    efficiency_data = {
        'Total Inference Time (seconds)': total_time,
        'Peak Memory Usage (MiB)': peak_memory,
        'Throughput (images/second)': throughput
    }

    for key, value in efficiency_data.items():
        print(f'{key}: {value}')

    # Optionally, save the data to a file
    with open('efficiency_report.txt', 'w') as f:
        for key, value in efficiency_data.items():
            f.write(f'{key}: {value}\\n')

    return efficiency_data#, results

# Example usage:
# results, _ = measure_efficiency(your_inference_function, your_data_batch)
if __name__ == '__main__':
    measure_efficiency(args)