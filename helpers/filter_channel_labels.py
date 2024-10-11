"""
This script processes a project configuration file and filters labeled anomalies
based on specific channels.

Steps:
1. Imports necessary functions from the `helpers` module.
2. Loads the project configuration from `prj_config.yaml`.
3. Retrieves project and run settings from the configuration.
4. Searches for the specified project within the configuration and retrieves its details.
5. Collects channel labels from the project configuration if the specified project is found.
6. Calls `filter_channel_labels` to filter anomalies from the input CSV file based on the
   collected channels and writes the results to an output CSV file.

Functions:
- `filter_channel_labels(INPUT_FILE, OUTPUT_FILE, channels)`: Filters anomalies in the
  `INPUT_FILE` based on the specified `channels` and writes the filtered results to `OUTPUT_FILE`.

Note:
- Ensure that the file paths for the input and output files are correctly specified.
- If the specified project is not found in the configuration, an error message is printed.
"""

import os
import sys
from helpers import (
    filter_channel_labels,
    get_projects_and_run_settings,
    load_config,
    check_chan_ids_exists
)

# Get channels from the project configuration
channels = []
prj_config = load_config('./prj_config.yaml')
example_projects, run_settings = get_projects_and_run_settings(prj_config)

example_project = None
for project in example_projects:
    if run_settings['project'] in project:
        example_project = project[run_settings['project']]

if example_project is None:
    print(f"[anomaly_aie_lstm] Project name {run_settings['project']}' not found in prj_config.yaml.")
else:
    for model_par in example_project['graphs']:
        for i in range(0, len(model_par['models'])):
            channels.append(model_par['models'][i])

    # Check channels against labeled_anomalies.csv to check if they are supported
    csv_file_path = '../telemanom/labeled_anomalies.csv'
    results = check_chan_ids_exists(csv_file_path, channels)
    for chan_id, exists in results.items():
        if not exists:
            print(f"[anomaly_aie_lstm] Channel ID {chan_id} is not supported.")
            sys.exit(1)

    INPUT_FILE = '../telemanom/labeled_anomalies.csv'
    OUTPUT_FILE = './filtered_labeled_anomalies.csv'

    filter_channel_labels(INPUT_FILE, OUTPUT_FILE, channels)
