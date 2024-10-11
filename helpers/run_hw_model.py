import os
import sys
from helpers import (
    load_config, 
    get_projects_and_run_settings, 
    execute_remote_script
)
sys.path.insert(0, '../telemanom')
from telemanom.helpers import Config

# Get Telemanom configuration
telemanom_config = Config('../telemanom/config.yaml')

# Get project configuration
prj_config = load_config('./prj_config.yaml')
projects, run_settings = get_projects_and_run_settings(prj_config)

local_script_path = os.path.join('../lstm_vek280', run_settings['project'], 'sw',
                                 'embedded_exec.sh')
remote_script_path = '/home/petalinux/embedded_exec.sh'  # Remote path where script will be uploaded
remote_output_dir = f'/run/media/mmcblk0p1/output'
local_output_dir = os.path.join('../lstm_vek280', run_settings['project'], 'output')

# Run model remotely on hardware
execute_remote_script(
    run_settings['ssh_hostname'],
    run_settings['ssh_port'],
    run_settings['ssh_username'],
    run_settings['ssh_password'],
    local_script_path,
    remote_script_path,
    remote_output_dir,
    local_output_dir
)
