import yaml
import os
import sys


def write_header_and_config_files(project_name, yaml_file, header_file, system_cfg_file_path):
    # Load the YAML config file
    try:
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        sys.exit(f"[anomaly_aie_lstm] Error loading config file: {e}")
    
    # Find the project in the example projects
    projects = config.get('example_projects', [])
    project_models = None

    for project in projects:
        if project_name in project:
            project_models = project[project_name].get('graphs', [])
            break
    
    # Error out if the project or models are not found
    if not project_models:
        sys.exit(f"[anomaly_aie_lstm] Error: Project '{project_name}' or its models not found.")

    # Write to the header file
    try:
        with open(header_file, 'w') as file:
            file.write("// Auto-generated file\n\n")
            file.write('#pragma once\n\n')

            file.write(f'// Models for project: {project_name}\n')

            num_graphs = len(project_models)
            max_models_per_graph = max(len(graph.get('models', [])) for graph in project_models)

            file.write(f'#define NUM_GRAPHS {num_graphs}\n')
            file.write(f'#define MAX_MODELS_PER_GRAPH {max_models_per_graph}\n\n')
            file.write('const char* PROJECT_MODELS[NUM_GRAPHS][MAX_MODELS_PER_GRAPH] = {\n')

            for graph in project_models:
                file.write('    { ')
                models = graph.get('models', [])

                for model in models:
                    file.write(f'"{model}", ')

                # Pad with nullptr if the number of models is less than max_models_per_graph
                for _ in range(max_models_per_graph - len(models)):
                    file.write('nullptr, ')

                file.write('},\n')

            file.write('};\n')

        print(f"[anomaly_aie_lstm] Successfully wrote header file")
    except Exception as e:
        sys.exit(f"[anomaly_aie_lstm] Error writing header file: {e}")

    # Generate the system.cfg content based on the number of graphs
    try:
        with open(system_cfg_file_path, 'w') as file:
            file.write("# Auto-generated file\n\n")
            file.write('[connectivity]\n')
            file.write(f'nk=mm2s:{num_graphs}:' + '.'.join(f'mm2s_{i}' for i in range(num_graphs)) + '\n')
            file.write(f'nk=s2mm:{num_graphs}:' + '.'.join(f's2mm_{i}' for i in range(num_graphs)) + '\n')
            
            for i in range(num_graphs):
                file.write(f'sc=mm2s_{i}.s:ai_engine_0.DataIn_{i}\n')
                file.write(f'sc=ai_engine_0.DataOut_{i}:s2mm_{i}.s\n')
        
        print(f"[anomaly_aie_lstm] system.cfg file generated at {system_cfg_file_path}")
    except Exception as e:
        sys.exit(f"[anomaly_aie_lstm] Error writing system.cfg file: {e}")


def main():
    if len(sys.argv) != 2:
        print("[anomaly_aie_lstm] Usage: python make_host_sw_header.py <project_name>")
        sys.exit(1)

    project_name = sys.argv[1]

    yaml_file = './prj_config.yaml'
    header_file = os.path.join('../lstm_vek280', project_name, 'sw/models.h')
    system_cfg_file_path = os.path.join('../lstm_vek280', project_name, 'system.cfg')

    try:
        write_header_and_config_files(project_name, yaml_file, header_file, system_cfg_file_path)
    except ValueError as e:
        print(f"[anomaly_aie_lstm] Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
