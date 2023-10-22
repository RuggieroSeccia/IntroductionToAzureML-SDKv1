import os
from pathlib import Path
from azureml.core import Workspace, Run

def fetch_uploaded_files_from_run(run_id: str,
                                  dir_data_to_fetch: str,
                                  output_file_path: str):
    """Function to fetch the files uploaded at run level. It returns the path where the data are downloaded"""
    os.makedirs(output_file_path, exist_ok=True)

    # we connect to the run
    ws = Workspace.from_config()
    run = Run.get(ws, run_id)

    # we download the data we want
    print(f"Downloading data from run with id {run_id}")
    run.download_file(dir_data_to_fetch, output_file_path=output_file_path)
    print(f'Data downloaded and saved at {output_file_path}')
    path_to_downloaded_data = os.path.join(output_file_path, os.path.basename(dir_data_to_fetch))
    return path_to_downloaded_data


def fetch_output_from_run_id(run_id: str,
                             data_to_fetch: str,
                             output_dir: Path):
    """Function to fetch the files in output from a Pipeline step"""

    saved_directory = os.path.join(output_dir, 'azureml', run_id, data_to_fetch)
    os.makedirs(saved_directory, exist_ok=True)

    ws = Workspace.from_config()
    run = Run.get(ws, run_id)
    run_output = run.get_outputs()
    try:
        step_run_output = run_output[data_to_fetch]
    except:
        raise ValueError(f"'{data_to_fetch}' not available. Outputs available are: {run_output.keys}")
    port_data_reference = step_run_output.get_port_data_reference()
    port_data_reference.download(local_path=output_dir)
    return saved_directory