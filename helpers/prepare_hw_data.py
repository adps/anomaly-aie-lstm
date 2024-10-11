"""
prepare_data.py

This script prepares the model and input data and configures settings for 
the embedded host application.

It performs the following tasks:

1. **Configuration Loading:**
   - Loads configurations for the Telemanom and example projects from YAML files.
   - Retrieves project and run settings from the example project configuration.

2. **Project Validation:**
   - Checks if the specified project in the run settings exists in the
     project configuration. If not, prints an error message and exits.

3. **Model Processing:**
   - For each model specified in the project configuration:
     - Converts model weights from HDF5 format to a format suitable for AIE-ML
       graphs.
     - Creates necessary directories for saving the converted model weights.
     - Handles errors during model conversion by inspecting the HDF5 file content
       and exits if an error occurs.

4. **Input Data Conversion:**
   - Converts test input data from NPY format to TXT format.
   - Saves the converted input data in the appropriate directory.

5. **Header File Modification:**
   - Updates the constants in the `config.h` header file to reflect the project
     settings and data specifications.
   - Modifies sample size, number of predictions, number of sensors, and other
     parameters as needed for AIE-ML compilation.

6. **Completion Message:**
   - Prints a message indicating the completion of data preparation for the
     specified project.

Dependencies:
- numpy: For handling and converting data arrays.
- os: For file and directory operations.
- sys: For exiting the script in case of errors.
- helpers: Custom helper functions for configuration loading, model conversion,
  and header modification.
- telemanom.telemanom.helpers: Custom helper functions and configuration
  management.

Usage:
    Run this script as a standalone program. Ensure that the required YAML
    configuration files are present in the specified paths.
"""

import os
import sys
import numpy as np
from helpers import (
    get_projects_and_run_settings,
    load_config,
    inspect_h5_file_content,
    convert_model_x2,
    update_values_in_config_header
)
sys.path.insert(0, '../telemanom')
from telemanom.helpers import Config

# Get Telemanom configuration
telemanom_config = Config('../telemanom/config.yaml')

# Get HW project configuration
prj_config = load_config('./prj_config.yaml')
example_projects, run_settings = get_projects_and_run_settings(prj_config)

example_project = None
for project in example_projects:
    if run_settings['project'] in project:
        example_project = project[run_settings['project']]

if example_project is None:
    print(f"[anomaly_aie_lstm] Project name '{run_settings['project']}' not found in prj_config.yaml.")
else:
    # Model stuffs
    model_folder_path = os.path.join('../telemanom/data', telemanom_config.use_id, 'models')
    project_dir = os.path.join('../lstm_vek280', run_settings['project'])

    num_sensors = []
    num_iterations = []

    for model_par in example_project['graphs']:
        for i in range(0, len(model_par['models'])):
            # print(model_par['models'][i])
            channel_id = model_par['models'][i]
            use_2buffs = "2buffs" in run_settings['project']

            # Convert model from HDF5 format to TXT file that can be consumed by the AIE-ML graphs
            h5_file_path = os.path.join(model_folder_path, channel_id + '.h5')
            model_hw_folder_path = os.path.join(project_dir, 'data', channel_id)
            os.makedirs(model_hw_folder_path, exist_ok=True)

            try:
                nsensor = convert_model_x2(h5_file_path, model_hw_folder_path, channel_id, use_2buffs)
                num_sensors.append(nsensor)
            except ValueError as e:
                inspect_h5_file_content(h5_file_path)
                print(f"[anomaly_aie_lstm] An error occurred (see inspection result above): {e}")
                sys.exit(1)

            # Convert input file from NPY to TXT
            print(f'[anomaly_aie_lstm] Converting test input data from NPY to TXT for {channel_id}')
            xin_file_path = os.path.join('../telemanom/data/test', f'{channel_id}.npy')
            xin_hw_file_path = os.path.join(project_dir, 'data', channel_id, 'xin.txt')

            xin = np.load(xin_file_path)
            xin_reshaped = xin.reshape(-1)
            if xin_reshaped.size % nsensor != 0:
                print(f'[anomaly_aie_lstm] Input size is not a multiple of {nsensor}')
                sys.exit(1)

            num_iterations.append(xin_reshaped.size)
            np.savetxt(xin_hw_file_path, xin_reshaped, fmt='%s')


    if not all(x == num_sensors[0] for x in num_sensors):
        print(f"[anomaly_aie_lstm] Error: Only models/channels with the same number of sensors are supported in the project '{run_settings['project']}'")
        print(f'[anomaly_aie_lstm] Found channels with the following number of sensors: {num_sensors}')
        sys.exit(1)

    num_iterations = [x // num_sensors[0] for x in num_iterations]
    print(f'[anomaly_aie_lstm] Iterations: {num_iterations}')

    # Modify constants in config.h header file for AIE-ML compilation
    print("[anomaly_aie_lstm] Modifying config.h...")
    header_file_path = os.path.join(project_dir, 'aie', 'config.h')
    sample_size = telemanom_config.l_s
    npredictions = telemanom_config.n_predictions
    nsensors = num_sensors[0]
    npar = len(example_project['graphs'])
    ntdm = len(example_project['graphs'][0]['models'])
    niter = max(num_iterations)

    update_values_in_config_header(
        header_file_path,
        sample_size,
        npredictions,
        nsensors,
        npar,
        ntdm,
        niter
    )

    print(f"[anomaly_aie_lstm] Data preparation completed for project '{run_settings['project']}'!")
    