import sys
import os 
sys.path.append(os.getcwd())
import subprocess
import csv
import fcntl
import json
import re


def parse_params(params):
    """Parse the command line parameters into a dictionary."""
    param_dict = {}
    param_parts = params.split(" ")
    for part in param_parts:
        if '=' in part:
            key, value = part.split('=', 1)
            param_dict[key.strip('--')] = value.strip("'\"")  # Remove quotes if any
    return param_dict

def parse_metrics(output):
    """Extract accuracy metrics from the output."""
    metrics = {}
    patterns = {
        "average_train_accuracy_train_data": r"average train accuracy on train data :\s+([\d.]+)",
        "average_test_accuracy_train_data": r"average test accuracy on train data:\s+([\d.]+)",
        "average_train_accuracy_train_data_quant": r"average train accuracy on train data after quantization:\s+([\d.]+)",
        "average_test_accuracy_train_data_quant": r"average test accuracy on train data after quantization:\s+([\d.]+)",
        "average_train_accuracy_test_data": r"average train accuracy on test data:\s+([\d.]+)",
        "average_test_accuracy_test_data": r"average test accuracy on test data:\s+([\d.]+)",
        "average_train_accuracy_test_data_quant": r"average train accuracy on test data after quantization:\s+([\d.]+)",
        "average_test_accuracy_test_data_quant": r"average test accuracy on test data after quantization:\s+([\d.]+)",
        "average_train_accuracy": r"average train accuracy:\s+([\d.]+)",
        "average test_accuracy": r"average test accuracy:\s+([\d.]+)",
        "average train_accuracy_quant": r"average train accuracy after quantization:\s+([\d.]+)",
        "average test_accuracy_quant": r"average test accuracy after quantization:\s+([\d.]+)",
        "global_message_len": r"global message_len:\s+([\d.]+)",
        "sum_task_message_lens": r"sum_task_message_lens:\s+([\d.]+)",
        "average_message_len": r"avarage single task message_len:\s+([\d.]+)",
        "global_part_bound": r"global part bound:\s+([\d.]+)",
        "task_specific_part_bound": r"task specific part bound:\s+([\d.]+)",
        "total_bound": r"total bound:\s+([\d.]+)",
        "empirical_err": r"empirical error:\s+([\d.]+)",
        #"std_test_bound": r"std for bound on test tasks:\s+([\d.]+)",
        "single_task_bound": r"avarage single task bound:\s+([\d.]+)",
        "catoni_bound": r"avarage single task catoni bound:\s+([\d.]+)",
        "multi_task_bound": r"multi_task_bound:\s+([\d.]+)"
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))
    return metrics

    

def read_and_remove_first_line(filename):
    with open(filename, 'r+') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        lines = file.readlines()
        if not lines:
            print("No more hyperparameters to process.")
            fcntl.flock(file, fcntl.LOCK_UN)
            return None
        first_line = lines[0].strip()
        file.seek(0)
        file.writelines(lines[1:])  # Write remaining lines back
        file.truncate()
        # Unlock the file
        fcntl.flock(file, fcntl.LOCK_UN)
    return first_line

def run_python_file_with_params(py_file, params):
    command = ["python", py_file] + params.split() 
    print(command) 
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {py_file}: {result.stderr}")
        return None
    return result.stdout.strip()

def append_to_csv(csv_file, data):
    """Append data to a CSV file."""
    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if file.tell() == 0:
            writer.writeheader()  # Write header only if file is empty
        writer.writerow(data)

def main(py_file, csv_file, params_file):
    print("here")
    params = read_and_remove_first_line(params_file)
    if params:
        output = run_python_file_with_params(py_file, params)
        print("finish running")
        if output:
            param_dict = parse_params(params)
            metrics = parse_metrics(output)
            combined_data = {**param_dict, **metrics}  
            append_to_csv(csv_file, combined_data)
            print("saved")
    return 0


def entrypoint(**kwargs):  
  main(**kwargs)


if __name__ == '__main__':
  import fire
  fire.Fire(entrypoint)
  