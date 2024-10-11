ifndef PROJECT_PYTHON_PATH
$(error PROJECT_PYTHON_PATH is not set. Please set it using: export PROJECT_PYTHON_PATH=/path/to/python3)
endif

PYTHON = $(PROJECT_PYTHON_PATH)
VENV := .venv
PIP = pip
HELPERS_DIR_NAME = helpers
TELEMANOM_REPO_URL = https://github.com/khundman/telemanom.git
TELEMANOM_DIR_NAME = telemanom
DATA_URL = https://s3-us-west-2.amazonaws.com/telemanom/data.zip
DATA_FILE = data.zip
DATA_DIR = $(TELEMANOM_DIR_NAME)/data
LSTM_VEK280_DIR_NAME = lstm_vek280
ALL_PROJECTS := $(shell ls $(LSTM_VEK280_DIR_NAME))
PROJECT := $(shell grep 'project:' $(HELPERS_DIR_NAME)/prj_config.yaml | sed 's/.*project: //')

.PHONY: \
    all \
    venv \
    print_project \
	clone_telemanom \
    download_and_unzip \
    prepare_hw_data \
	generate_config \
	build_project \
	run_sw \
	run_hw \
	visualize_sw \
	visualize_hw \
	clean_data \
	clean_hw \
    clean_all

all: print_project venv clone_telemanom download_and_unzip prepare_hw_data generate_config build_project

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	./$(VENV)/bin/$(PIP) install --upgrade pip
	./$(VENV)/bin/$(PIP) install -r requirements.txt
	./$(VENV)/bin/$(PIP) install --upgrade nbformat

print_project:
	@echo "All projects: $(ALL_PROJECTS)"
	@echo "Project: $(PROJECT)"

venv: $(VENV)/bin/activate

clone_telemanom: venv
	@if [ ! -d "$(TELEMANOM_DIR_NAME)" ]; then \
		echo "[anomaly_aie_lstm] Cloning the Telemanom repo from $(TELEMANOM_REPO_URL)..."; \
		git clone $(TELEMANOM_REPO_URL); \
		./$(VENV)/bin/$(PIP) install -r $(TELEMANOM_DIR_NAME)/requirements.txt; \
	else \
		echo "[anomaly_aie_lstm] 'telemanom' directory already exists, skipping clone."; \
	fi

download_and_unzip: clone_telemanom
	@if [ ! -d "$(DATA_DIR)" ]; then \
		echo "[anomaly_aie_lstm] Downloading data from $(DATA_URL)..."; \
		curl -L $(DATA_URL) -o $(DATA_FILE); \
		unzip -o $(DATA_FILE) -d $(TELEMANOM_DIR_NAME); \
		rm -f $(DATA_FILE); \
	else \
		echo "[anomaly_aie_lstm] Data directory already exists, skipping download and unzip."; \
	fi

prepare_hw_data: venv download_and_unzip
	@if [ ! -d "$(LSTM_VEK280_DIR_NAME)/$(PROJECT)/data" ]; then \
		echo "[anomaly_aie_lstm] Running prepare_hw_data.py..."; \
		cd $(HELPERS_DIR_NAME) && ../$(VENV)/bin/python3 prepare_hw_data.py; \
	else \
		echo "[anomaly_aie_lstm] Hardware data directory already exists, run 'make clean_data' to remove it first."; \
	fi

generate_config: venv
	@echo "[anomaly_aie_lstm] Running generate_config.py..."
	@cd $(HELPERS_DIR_NAME) && ../$(VENV)/bin/python3 generate_config.py $(PROJECT)

build_project: venv prepare_hw_data generate_config
	@echo "[anomaly_aie_lstm] Compiling the project $(PROJECT)..."
	make -C $(LSTM_VEK280_DIR_NAME)/$(PROJECT) aie || exit 1
	make -C $(LSTM_VEK280_DIR_NAME)/$(PROJECT) kernels || exit 1
	make -C $(LSTM_VEK280_DIR_NAME)/$(PROJECT) xsa || exit 1
	make -C $(LSTM_VEK280_DIR_NAME)/$(PROJECT) host || exit 1
	make -C $(LSTM_VEK280_DIR_NAME)/$(PROJECT) package || exit 1

run_sw: venv download_and_unzip
	@echo "[anomaly_aie_lstm] Running the SW model on the local CPU..."
	@cd $(HELPERS_DIR_NAME) && ../$(VENV)/bin/python3 filter_channel_labels.py && \
	echo "[anomaly_aie_lstm] Running example.py from the $(TELEMANOM_DIR_NAME) folder..." && \
	cd ../$(TELEMANOM_DIR_NAME) && ../$(VENV)/bin/python3 example.py -l ../$(HELPERS_DIR_NAME)/filtered_labeled_anomalies.csv

run_hw: venv prepare_hw_data
	@echo "[anomaly_aie_lstm] Running the HW model on the VEK280..."
	@cd $(HELPERS_DIR_NAME) && ../$(VENV)/bin/python3 run_hw_model.py

visualize_sw: venv
	@echo "[anomaly_aie_lstm] Copying result-viewer-sw.ipynb to $(TELEMANOM_DIR_NAME)/telemanom/..."
	@cp $(HELPERS_DIR_NAME)/result-viewer-sw.ipynb $(TELEMANOM_DIR_NAME)/telemanom/
	@echo "[anomaly_aie_lstm] Launching Jupyter Notebook for SW result visualization..."
	./$(VENV)/bin/jupyter notebook $(TELEMANOM_DIR_NAME)/telemanom/result-viewer-sw.ipynb

visualize_hw: venv
	@echo "[anomaly_aie_lstm] Copying result-viewer-hw.ipynb to $(TELEMANOM_DIR_NAME)/telemanom/..."
	@cp $(HELPERS_DIR_NAME)/result-viewer-hw.ipynb $(TELEMANOM_DIR_NAME)/telemanom/
	@echo "[anomaly_aie_lstm] Launching Jupyter Notebook for HW result visualization..."
	./$(VENV)/bin/jupyter notebook $(TELEMANOM_DIR_NAME)/telemanom/result-viewer-hw.ipynb

clean_data:
	rm -rf $(LSTM_VEK280_DIR_NAME)/$(PROJECT)/data

clean_hw:
	make -C $(LSTM_VEK280_DIR_NAME)/$(PROJECT) clean

clean_all:
	rm -rf "$(VENV)" $(HELPERS_DIR_NAME)/__pycache__
	rm -f $(HELPERS_DIR_NAME)/filtered_labeled_anomalies.csv
	rm -f "$(DATA_FILE)"
	rm -rf "$(TELEMANOM_DIR_NAME)"
	find $(LSTM_VEK280_DIR_NAME) -type d -name 'data' -exec rm -rf {} +
	@for dir in $(ALL_PROJECTS); do \
        echo "Cleaning $$dir..."; \
        make -C $(LSTM_VEK280_DIR_NAME)/$$dir clean; \
    done
