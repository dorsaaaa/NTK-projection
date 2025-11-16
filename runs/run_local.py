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



def main(py_file, csv_file, params_file, array_len = 1):
    
    inputs = {'py_file': py_file, 'csv_file': csv_file, 'params_file':params_file}
    input_sets = [inputs]*array_len
    available_gpus = [0]  
    
    # Create a list to hold futures
    futures = []

    # Use ProcessPoolExecutor to run tasks in parallel
    with ProcessPoolExecutor(max_workers=1) as executor:
        for run_id, inputs in enumerate(input_sets, start=1):
            gpu_id = available_gpus[0]
            # Submit the task to the executor
            print(f"Run {run_id} on gpu {gpu_id} started.")
            future = executor.submit(run_project, inputs, 'run.py', gpu_id)
            futures.append((run_id, future))
        executor.shutdown(wait=True)
    
     # Collect results as they complete
        for run_id, future in futures:
            result = future.result()  # Blocking call until the result is available
            gpu_id = available_gpus[0]
            print(f"Run {run_id} on gpu {gpu_id} completed and results saved.")




def entrypoint(**kwargs):  
  main(**kwargs)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    import fire
    fire.Fire(entrypoint)