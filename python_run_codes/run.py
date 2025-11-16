import csv
import subprocess
import sys
import os
from contextlib import contextmanager
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
import os 
sys.path.append(os.getcwd())

@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def run_project(input_params,file_name,gpu_id):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    spec = importlib.util.spec_from_file_location("main", file_name)
    project_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_module)
    with suppress_stdout_stderr():
        result = project_module.main(**input_params)  
    return result


def save_results_to_csv(filename, data):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

def main(mode='single'):
    if mode=='meta':
        file_name = 'experiments/train.py'
        csv_filename = 'MetaSAID_results_shuffled_labels.csv' 
        header = ['epochs', 'intrinsic_dim','lr', 'seed', 'is_said', 'quantize_type', 'quant_epochs', 'gpu_id', 'avg training train_acc', 'avg training test_acc', 'avg training train_acc_quant', 'avg training test_acc_quant', 'avg testing train_acc', 'avg testing test_acc', 'avg quantized train_acc', 'avg quantized test_acc']
    else:
        file_name = 'experiments/train_single_task.py'
        csv_filename = 'single_task_results_diff_sample.csv' 
        header = ['data_seed', 'num_samples', 'epochs', 'intrinsic_dim','lr', 'seed', 'is_said', 'quantize_type', 'quant_epochs', 'train_acc', 'test_acc', 'quantized_train_acc', 'quantized_test_acc']

    
    '''
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
    '''
    if not os.path.exists(csv_filename):
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
    input_sets = [
        #{'epochs':100, 'intrinsic_dim':200,'lr':0.1, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':300,'lr':0.1, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':400,'lr':0.1, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':500,'lr':0.1, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':600,'lr':0.1, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':200,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':400,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':500,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},

        #{'epochs':200, 'intrinsic_dim':200,'lr':0.1, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':200, 'intrinsic_dim':300,'lr':0.1, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':200, 'intrinsic_dim':400,'lr':0.1, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':200, 'intrinsic_dim':500,'lr':0.1, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':200, 'intrinsic_dim':600,'lr':0.1, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':200, 'intrinsic_dim':200,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':200, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':200, 'intrinsic_dim':400,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':200, 'intrinsic_dim':500,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':200, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
                                 
        #{'epochs':400, 'intrinsic_dim':200,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':400, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':400, 'intrinsic_dim':400,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':400, 'intrinsic_dim':500,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':400, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':400, 'intrinsic_dim':200,'lr':0.001, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':400, 'intrinsic_dim':300,'lr':0.001, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':400, 'intrinsic_dim':400,'lr':0.001, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':400, 'intrinsic_dim':500,'lr':0.001, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':400, 'intrinsic_dim':600,'lr':0.001, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},  

        #{'epochs':100, 'intrinsic_dim':200,'lr':0.1, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':300,'lr':0.1, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':400,'lr':0.1, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':500,'lr':0.1, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':600,'lr':0.1, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':200,'lr':0.01, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':400,'lr':0.01, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':500,'lr':0.01, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
    
        #{'epochs':200, 'intrinsic_dim':200,'lr':0.1, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':200, 'intrinsic_dim':300,'lr':0.1, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':200, 'intrinsic_dim':400,'lr':0.1, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':200, 'intrinsic_dim':500,'lr':0.1, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':200, 'intrinsic_dim':600,'lr':0.1, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        #{'epochs':200, 'intrinsic_dim':200,'lr':0.01, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        {'epochs':200, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        {'epochs':200, 'intrinsic_dim':400,'lr':0.01, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        {'epochs':200, 'intrinsic_dim':500,'lr':0.01, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        {'epochs':200, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        
        {'epochs':400, 'intrinsic_dim':200,'lr':0.01, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        {'epochs':400, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        {'epochs':400, 'intrinsic_dim':400,'lr':0.01, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        {'epochs':400, 'intrinsic_dim':500,'lr':0.01, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        {'epochs':400, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        {'epochs':400, 'intrinsic_dim':200,'lr':0.001, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        {'epochs':400, 'intrinsic_dim':300,'lr':0.001, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        {'epochs':400, 'intrinsic_dim':400,'lr':0.001, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        {'epochs':400, 'intrinsic_dim':500,'lr':0.001, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30},
        {'epochs':400, 'intrinsic_dim':600,'lr':0.001, 'seed':137, 'is_said':True, 'quantize_type':'default', 'quant_epochs':30}  

        #{'data_seed':42 , 'num_samples': 100, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':42 , 'num_samples': 200, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':42 , 'num_samples': 300, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':42 , 'num_samples': 400, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':42 , 'num_samples': 500, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':42 , 'num_samples': 600, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},

        #{'data_seed':142 ,'num_samples': 100, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':142 ,'num_samples': 200, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':142 ,'num_samples': 300, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':142 ,'num_samples': 400, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':142 ,'num_samples': 500, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':142 ,'num_samples': 600, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},

        #{'data_seed':242 ,'num_samples': 100, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':242 ,'num_samples': 200, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':242 ,'num_samples': 300, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':242 ,'num_samples': 400, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':242 ,'num_samples': 500, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':242 ,'num_samples': 600, 'epochs':100, 'intrinsic_dim':300,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},

        #{'data_seed':42 , 'num_samples': 100, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':42 , 'num_samples': 200, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':42 , 'num_samples': 300, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':42 , 'num_samples': 400, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':42 , 'num_samples': 500, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':42 , 'num_samples': 600, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
    
        #{'data_seed':142 ,'num_samples': 100, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':142 ,'num_samples': 200, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':142 ,'num_samples': 300, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':142 ,'num_samples': 400, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':142 ,'num_samples': 500, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':142 ,'num_samples': 600, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},

        #{'data_seed':242 ,'num_samples': 100, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':242 ,'num_samples': 200, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':242 ,'num_samples': 300, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':242 ,'num_samples': 400, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':242 ,'num_samples': 500, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30},
        #{'data_seed':242 ,'num_samples': 600, 'epochs':100, 'intrinsic_dim':600,'lr':0.01, 'seed':137, 'is_said':False, 'quantize_type':'default', 'quant_epochs':30}    
    
    ]
    available_gpus = [0,1,2,3,4,5,6,7]  

    # Create a list to hold futures
    futures = []

    # Use ProcessPoolExecutor to run tasks in parallel
    with ProcessPoolExecutor(max_workers=len(available_gpus)) as executor:
        for run_id, inputs in enumerate(input_sets, start=1):
            gpu_id = available_gpus[run_id % len(available_gpus)]
            # Submit the task to the executor
            print(f"Run {run_id} on gpu {gpu_id} started.")
            inputs['device_id']=gpu_id
            future = executor.submit(run_project, inputs, file_name, gpu_id)
            futures.append((run_id, future))
    
     # Collect results as they complete
        for run_id, future in futures:
            result = future.result()  # Blocking call until the result is available
            gpu_id = available_gpus[run_id % len(available_gpus)]
            data_to_save = list(input_sets[run_id-1].values())+result
            save_results_to_csv(csv_filename, data_to_save)
            print(f"Run {run_id} on gpu {gpu_id} completed and results saved.")


def entrypoint(**kwargs):  
  main(**kwargs)


if __name__ == '__main__':
  import fire
  fire.Fire(entrypoint)