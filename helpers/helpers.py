import os
import numpy as np
import h5py
import paramiko
import re
import yaml
import csv
import pandas as pd

# Function to load YAML file
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config


# Function to retrieve projects and run_settings
def get_projects_and_run_settings(config):
    projects = config.get('example_projects', [])
    run_settings = config.get('run_settings', {})
    return projects, run_settings


def check_chan_ids_exists(csv_file_path, chan_ids):
    """
    Check if the given array of chan_ids exists in the specified CSV file.

    :param csv_file_path: Path to the CSV file.
    :param chan_ids: List of channel IDs to search for.
    :return: A dictionary with chan_ids as keys and booleans as values indicating existence.
    """
    chan_ids_set = set(chan_ids)  # Convert to set for faster lookup
    found_chan_ids = {chan_id: False for chan_id in chan_ids}
    
    with open(csv_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['chan_id'] in chan_ids_set:
                found_chan_ids[row['chan_id']] = True
                chan_ids_set.remove(row['chan_id'])
                if not chan_ids_set:
                    break
    
    return found_chan_ids


def filter_channel_labels(input_file_path, output_file_path, channels_to_include):
    """
    Filters rows from the input CSV file based on the specified
    channels and writes the filtered data to a new CSV file.

    Args:
        input_file_path (str): Path to the input CSV file.
        output_file_path (str): Path to the output CSV file.
        channels_to_include (list): List of channel IDs to include in the output.
    """
    filtered_rows = []

    # Open and read the input CSV file
    with open(input_file_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        headers = reader.fieldnames  # Get the headers from the input file

        # Iterate over each row in the CSV
        for row in reader:
            if row['chan_id'] in channels_to_include:
                filtered_rows.append(row)

    # Open and write to the output CSV file
    with open(output_file_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()  # Write the header row
        writer.writerows(filtered_rows)  # Write the data rows


def txt_to_npy_selective_reshape(input_txt_file, output_npy_file, nth_value_to_read, shape):
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_npy_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(input_txt_file, 'r') as file:
        float_values = [float(line.strip()) for i, line in enumerate(file) if i % nth_value_to_read == 0]

    np_array = np.array(float_values)
    np_array = np_array.reshape(shape)
    np.save(output_npy_file, np_array)


def update_values_in_config_header(file_path, sample_size_value, npredictions_value,
                                   nsensors_value, npar_value, ntdm_value, niter_value):
    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Define the regular expressions to find the specific macros
    sample_size_pattern = r"#define SAMPLE_SIZE (\d+)"
    npredictions_pattern = r"#define NPREDICTIONS (\d+)"
    nsensors_pattern = r"#define NSENSORS (\d+)"
    npar_pattern = r"#define NPAR (\d+)"
    ntdm_pattern = r"#define NTDM (\d+)"
    niter_pattern = r"#define NITERATIONS (\d+)"

    # Modify the lines that contain SAMPLE_SIZE, NPREDICTIONS, NSENSORS, and NTDM
    with open(file_path, 'w') as file:
        for line in content:
            if re.match(sample_size_pattern, line):
                file.write(f"#define SAMPLE_SIZE {sample_size_value} // number of previous timesteps provided to model to predict future values\n")
            elif re.match(npredictions_pattern, line):
                file.write(f"#define NPREDICTIONS {npredictions_value} // number of steps ahead to predict\n")
            elif re.match(nsensors_pattern, line):
                file.write(f"#define NSENSORS {nsensors_value} // number of sensors\n")
            elif re.match(npar_pattern, line):
                file.write(f"#define NPAR {npar_value} // number of graphs in parallel\n")
            elif re.match(ntdm_pattern, line):
                file.write(f"#define NTDM {ntdm_value} // number of models time-multiplexed in each graph\n")
            elif re.match(niter_pattern, line):
                file.write(f"#define NITERATIONS {niter_value} // number of graph iterations\n")
            else:
                file.write(line)


def inspect_h5_file_content(h5_file_path):
    """Function to inspect an HDF5 file and print its structure."""
    try:
        with h5py.File(h5_file_path, 'r') as h5_file:
            print(f"[anomaly_aie_lstm] Inspecting HDF5 file: {h5_file_path}")
            print("-" * 60)

            def print_structure(name, obj):
                """Helper function to recursively print the structure of the file."""
                if isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name}, Shape: {obj.shape}, Type: {obj.dtype}")
                elif isinstance(obj, h5py.Group):
                    print(f"Group: {name}")

            # Visit every object in the file
            h5_file.visititems(print_structure)

            print("-" * 60)

    except Exception as e:
        print(f"Error while inspecting HDF5 file: {e}")


def convert_model_x2(h5_file_path, data_hw_folder_path, channel_id, use_2buffs):
    with h5py.File(h5_file_path, 'r') as f:
        # Access the 'model_weights' group
        model_weights = f['model_weights']

        # List all groups in 'model_weights'
        groups = list(model_weights.keys())

        # Helper function to extract the number after 'lstm_' or 'dense_'
        def extract_number(group_name):
            match = re.search(r'_(\d+)', group_name)
            return int(match.group(1)) if match else None

        # Identify LSTM and Dense layers and sort them by the numerical suffix
        lstm_groups = sorted([name for name in groups if 'lstm' in name],
                             key=lambda x: extract_number(x))
        dense_groups = sorted([name for name in groups if 'dense' in name],
                              key=lambda x: extract_number(x))

        # Ensure there are enough LSTM and Dense layers
        if len(lstm_groups) < 2 or len(dense_groups) < 1:
            raise ValueError('Not enough LSTM or Dense layers found.')

        # Select the first two LSTM layers and the first Dense layer
        lstm1_group = model_weights[f"{lstm_groups[0]}/{lstm_groups[0]}"]
        lstm2_group = model_weights[f"{lstm_groups[1]}/{lstm_groups[1]}"]
        dense_group = model_weights[f"{dense_groups[0]}/{dense_groups[0]}"]

        # Get the shape of the LSTM 1st layer weights
        kernel_0 = lstm1_group['kernel:0'][:]
        c, r = kernel_0.shape

        # Check if model size is as expected
        if r != 320:
            raise ValueError(f'[anomaly_aie_lstm] Model {channel_id} size not as expected, incorrect number of neurons: {r}')
        if c not in [25, 55]:
            raise ValueError(f'[anomaly_aie_lstm] Model {channel_id} size not as expected, incorrect number of inputs: {c}')

        print(f"[anomaly_aie_lstm] Converting model with {c} inputs")

        # Converting LSTM 1st layer
        print(f"[anomaly_aie_lstm] Converting LSTM 1st Layer with {c} x {r} Input Weights")
        x1 = lstm1_group['kernel:0'][:].reshape(-1)
        print(f"[anomaly_aie_lstm] Converting LSTM 1st Layer with {r//4} x {r} Recurrent Weights")
        x2 = lstm1_group['recurrent_kernel:0'][:].reshape(-1)
        print(f"[anomaly_aie_lstm] Converting LSTM 1st Layer with {r} Bias Weights")
        x3 = lstm1_group['bias:0'][:].reshape(-1)

        x = np.concatenate([x1, x2, x3])

        # Write to file for LSTM 1st layer
        if use_2buffs:
            # Split the data into two parts: first 16,960 values in part1 and the rest in part2
            # NUM_NEURONS*4*(NUM_FEATURES+NUM_NEURONS+1+PADDING)/2
            part1_size = ((c + r//4 + 1 + 0) * 320) // 2

            # Part 1
            fstring = os.path.join(data_hw_folder_path, 'lstm1_x2fp_part1.txt')
            print(f"[anomaly_aie_lstm] Writing LSTM 2nd Layer (Part 1) to file {fstring}")
            with open(fstring, 'w') as fid:
                for k in range(0, part1_size, 2):
                    fid.write(f"{x[k]:2.8f} {x[k+1]:2.8f}\n")

            # Part 2 (rest of the values)
            fstring = os.path.join(data_hw_folder_path, 'lstm1_x2fp_part2.txt')
            print(f"[anomaly_aie_lstm] Writing LSTM 2nd Layer (Part 2) to file {fstring}")
            with open(fstring, 'w') as fid:
                for k in range(part1_size, len(x), 2):
                    fid.write(f"{x[k]:2.8f} {x[k+1]:2.8f}\n")
        else:
            fstring = os.path.join(data_hw_folder_path, 'lstm1_x2fp.txt')
            print(f"[anomaly_aie_lstm] Writing LSTM 1st Layer to file {fstring}")
            with open(fstring, 'w') as fid:
                for k in range(0, len(x), 2):
                    fid.write(f"{x[k]:2.8f} {x[k+1]:2.8f}\n")

        # Converting LSTM 2nd layer
        print(f"[anomaly_aie_lstm] Converting LSTM 2nd Layer with {r//4} x {r} Input Weights")
        x1 = lstm2_group['kernel:0'][:].reshape(-1)
        print(f"[anomaly_aie_lstm] Converting LSTM 2nd Layer with {r//4} x {r} Recurrent Weights")
        x2 = lstm2_group['recurrent_kernel:0'][:].reshape(-1)
        print(f"[anomaly_aie_lstm] Converting LSTM 2nd Layer with {r} Bias Weights")
        x3 = lstm2_group['bias:0'][:].reshape(-1)

        x = np.concatenate([x1, x2, x3])

        # Write to file for LSTM 2nd layer
        if use_2buffs:
            # Split the data into two parts: first 25,920 values in part1 and the rest in part2
            # NUM_NEURONS*4*(NUM_FEATURES+NUM_NEURONS+1+PADDING)/2
            part1_size = ((r//4 + r//4 + 1 + 1) * 320) // 2

            # Part 1
            fstring = os.path.join(data_hw_folder_path, 'lstm2_x2fp_part1.txt')
            print(f"[anomaly_aie_lstm] Writing LSTM 2nd Layer (Part 1) to file {fstring}")
            with open(fstring, 'w') as fid:
                for k in range(0, part1_size, 2):
                    fid.write(f"{x[k]:2.8f} {x[k+1]:2.8f}\n")

            # Part 2 (rest of the values)
            fstring = os.path.join(data_hw_folder_path, 'lstm2_x2fp_part2.txt')
            print(f"[anomaly_aie_lstm] Writing LSTM 2nd Layer (Part 2) to file {fstring}")
            with open(fstring, 'w') as fid:
                for k in range(part1_size, len(x), 2):
                    fid.write(f"{x[k]:2.8f} {x[k+1]:2.8f}\n")
        else:
            fstring = os.path.join(data_hw_folder_path, 'lstm2_x2fp.txt')
            print(f"[anomaly_aie_lstm] Writing LSTM 2nd Layer to file {fstring}")
            with open(fstring, 'w') as fid:
                for k in range(0, len(x), 2):
                    fid.write(f"{x[k]:2.8f} {x[k+1]:2.8f}\n")

        # Converting Dense Layer
        print(f'[anomaly_aie_lstm] Converting Dense Layer with {r // 4} x 10 Input Weights: padding with zeros to 16 neurons')
        x1 = dense_group['kernel:0'][()]  # Shape (80, 10)
        # Pad to match 16 neurons
        padding_size = 16 - x1.shape[1]
        if padding_size > 0:
            x1 = np.hstack([x1, np.zeros((x1.shape[0], padding_size))])
        x1 = np.reshape(x1, (r // 4 * 16, 1))
        print(f'[anomaly_aie_lstm] Converting Dense Layer with 10 Bias Weights: padding with zeros to 16 neurons')
        x2 = dense_group['bias:0'][()]  # Shape (10,)
        x2 = np.concatenate([x2, np.zeros(6)]).reshape(-1, 1)  # Padding with zeros to match 16 neurons
        # Combine Dense layer weights
        x = np.concatenate([x1, x2])

        # Write to file for Dense layer
        fstring = os.path.join(data_hw_folder_path, 'dense_x2fp.txt')
        print(f"[anomaly_aie_lstm] Writing Dense Layer to file {fstring}")
        with open(fstring, 'w') as fid:
            for k in range(0, len(x), 2):
                fid.write(f'{x[k][0]:2.8f} {x[k+1][0]:2.8f}\n')
        
        return c


def execute_remote_script(hostname, port, username, password, local_script_path,
                          remote_script_path, remote_output_dir, local_output_dir):
    try:
        # Create SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, port=port, username=username, password=password)

        # Create SFTP client for file transfer
        sftp = ssh.open_sftp()

        # Ensure the target directory exists
        remote_dir = '/'.join(remote_script_path.split('/')[:-1])
        try:
            sftp.chdir(remote_dir)  # Change to the remote directory to check if it exists
        except FileNotFoundError:
            print(f"[anomaly_aie_lstm] Target directory {remote_dir} does not exist. Creating it.")
            sftp.makedirs(remote_dir)  # Create the directory if it does not exist

        # Upload the script
        try:
            sftp.put(local_script_path, remote_script_path)
            print(f"[anomaly_aie_lstm] Uploaded script to {remote_script_path}")
        except IOError as e:
            print(f"[anomaly_aie_lstm] Permission error or file upload failed: {e}")
            return

        # Make the script executable
        ssh.exec_command(f'chmod +x {remote_script_path}')

        # Run the script using sudo and pass the password via stdin
        sudo_command = f"echo '{password}' | sudo -S bash {remote_script_path}"
        stdin, stdout, stderr = ssh.exec_command(sudo_command)
        stdout.channel.recv_exit_status()  # Ensure the command has finished

        # Print command output and errors
        print("STDOUT:")
        print(stdout.read().decode())

        print("STDERR:")
        print(stderr.read().decode())

        # Check for errors in the script execution
        error_output = stderr.read().decode()
        if error_output:
            print(f"[anomaly_aie_lstm] Error during script execution: {error_output}")
            return

        # Check if the remote output directory exists
        try:
            sftp.chdir(remote_output_dir)
        except FileNotFoundError:
            print(f"[anomaly_aie_lstm] Remote output directory {remote_output_dir} does not exist.")
            return

        # List all files in the remote directory
        files = sftp.listdir(remote_output_dir)

        # Ensure local output directory exists
        if not os.path.exists(local_output_dir):
            os.makedirs(local_output_dir)

        # Download each file from the remote directory to the local directory
        for file in files:
            remote_file_path = os.path.join(remote_output_dir, file)
            local_file_path = os.path.join(local_output_dir, file)
            sftp.get(remote_file_path, local_file_path)
            print(f"[anomaly_aie_lstm] Copied {remote_file_path} to {local_file_path}")

        # Close SFTP client
        sftp.close()

    except Exception as e:
        print(f"[anomaly_aie_lstm] An error occurred: {e}")
    finally:
        # Close SSH connection
        ssh.close()

def plot_channel_hw(self, channel_id, plot_train=False, plot_errors=True):
        """
        Adapted from telemanom/telemanom/plotting.py. Generate interactive 
        plots for a channel using results from hardware run.
        By default it prints actual and predicted telemetry values.

        Args:
            channel_id (str): channel id
            plot_train (bool): If true, plot training data in separate plot
            plot_errors (bool): If true, plot prediction errors in separate plot
        """
        channel = self.result_df[self.result_df['chan_id'] == channel_id]

        plot_values = {
            'y_hat': np.load(os.path.join('..', 'data', self.run_id, 'y_hat_hw',
                                        '{}.npy'.format(channel_id))),
            'smoothed_errors': np.load(os.path.join('..', 'data', self.run_id,
                                                    'smoothed_errors_hw',
                                                    '{}.npy'.format(channel_id))),
            'test': np.load(os.path.join('..', 'data', 'test', '{}.npy'
                                        .format(channel_id))),
            'train': np.load(os.path.join('..', 'data', 'train', '{}.npy'
                                        .format(channel_id)))
        }

        self.channel_result_summary(channel, plot_values)

        sequence_type = 'true' if self.labels_available else 'predicted'
        y_shapes = self.create_shapes(eval(channel['anomaly_sequences'].values[0]),
                                      sequence_type, -1, 1, plot_values)
        e_shapes = self.create_shapes(eval(channel['anomaly_sequences'].values[0]),
                                      sequence_type, 0, None, plot_values)

        if self.labels_available:
            y_shapes += self.create_shapes(eval(channel['tp_sequences'].values[0])
                                           + eval(channel['fp_sequences'].values[0]),
                                           'predicted', -1, 1, plot_values)
            e_shapes += self.create_shapes(eval(channel['tp_sequences'].values[0])
                                           + eval(channel['fp_sequences'].values[0]),
                                           'predicted', 0, None, plot_values)

        train_df = pd.DataFrame({
            'train': plot_values['train'][:,0]
        })

        y_df = pd.DataFrame({
            'y_hat': plot_values['y_hat'].reshape(-1,)
        })

        y = plot_values['test'][self.config.l_s:-self.config.n_predictions][:,0]
        y_df['y'] = y
        if not len(y) == len(plot_values['y_hat']):
            modified_l_s = len(plot_values['y_test']) \
                           - len(plot_values['y_hat']) - 1
            y_df['y'] = y[modified_l_s:-1]

        e_df = pd.DataFrame({
            'e_s': plot_values['smoothed_errors'].reshape(-1,)
        })

        y_layout = {
            'title': 'y / y_hat comparison',
            'shapes': y_shapes,
        }

        e_layout = {
            'title': "Smoothed Errors (e_s)",
            'shapes': e_shapes,
        }

        if plot_train:
            train_df.iplot(kind='scatter', color='green',
                           layout={'title': "Training Data"})

        y_df.iplot(kind='scatter', layout=y_layout)

        if plot_errors:
            e_df.iplot(kind='scatter', layout=e_layout, color='red')
