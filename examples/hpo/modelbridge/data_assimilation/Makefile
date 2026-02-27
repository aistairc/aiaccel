SHELL := /bin/bash

THIS_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT ?= $(shell git -C "$(THIS_DIR)" rev-parse --show-toplevel 2>/dev/null || (cd "$(THIS_DIR)/../../../.." && pwd))

PYTHON ?= python
CONFIG_FILE ?= $(THIS_DIR)/mas_bench_config.yaml
CONFIG_PATH := $(abspath $(CONFIG_FILE))

OUTPUT_ROOT ?= $(REPO_ROOT)/work/modelbridge/data_assimilation
STATE_DIR := $(OUTPUT_ROOT)/state
LOG_DIR := $(OUTPUT_ROOT)/logs
MPLCONFIGDIR ?= $(OUTPUT_ROOT)/.mplconfig

AIACCEL_CMD ?= aiaccel-job local cpu $(LOG_DIR)/aiaccel_job_da.log

export MPLCONFIGDIR
export PYTHONPATH ?= $(REPO_ROOT)

.PHONY: all run clean

all: run

run: $(STATE_DIR)/01_data_assimilation.done

$(STATE_DIR):
	@mkdir -p "$(STATE_DIR)"
	@mkdir -p "$(LOG_DIR)"
	@mkdir -p "$(MPLCONFIGDIR)"

$(STATE_DIR)/01_data_assimilation.done: $(CONFIG_PATH) | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Data Assimilation] Run MAS-Bench Wrapper"
	@echo "========================================="
	cd "$(REPO_ROOT)" && $(AIACCEL_CMD) -- \
		"$(PYTHON)" "$(THIS_DIR)/mas_bench_wrapper.py" \
			--config "$(CONFIG_PATH)" \
			--output-root "$(OUTPUT_ROOT)"
	touch "$@"

clean:
	@echo "Cleaning data assimilation workspace..."
	@rm -rf "$(OUTPUT_ROOT)"
