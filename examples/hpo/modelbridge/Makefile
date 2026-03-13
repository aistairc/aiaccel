SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

EXAMPLE_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PROJECT_ROOT := $(abspath $(EXAMPLE_DIR)/../../..)
MODELBRIDGE_APP := $(PROJECT_ROOT)/aiaccel/hpo/apps/modelbridge.py

resolve_path = $(if $(filter /%,$(1)),$(1),$(EXAMPLE_DIR)/$(1))

# Directories
WORKSPACE_DIR ?= workspace
CONFIG_DIR := config

# Default configuration file
CONFIG_FILE ?= $(CONFIG_DIR)/config.yaml
WORKSPACE_PATH := $(call resolve_path,$(WORKSPACE_DIR))
CONFIG_PATH := $(call resolve_path,$(CONFIG_FILE))
STATE_DIR := $(WORKSPACE_PATH)/state
LOGS_DIR := $(WORKSPACE_PATH)/logs

# Shared runtime options
PYTHON ?= python
MPLCONFIGDIR ?= $(WORKSPACE_PATH)/.mplconfig

# Export variables for modelbridge tools
export CONFIG_FILE := $(CONFIG_PATH)
export LOGS_DIR
export MODELBRIDGE_CONFIG_FILE := $(CONFIG_PATH)
export MPLCONFIGDIR
export PYTHON
export STATE_DIR
export WORKSPACE_DIR := $(WORKSPACE_PATH)

.PHONY: all clean prepare hpo-train hpo-test collect fit evaluate

all: evaluate
	@echo "========================================="
	@echo "  Modelbridge Pipeline Completed Successfully"
	@echo "========================================="

prepare: $(STATE_DIR)/01_prepare.done
hpo-train: $(STATE_DIR)/02_hpo_train.done
hpo-test: $(STATE_DIR)/03_hpo_test.done
collect: $(STATE_DIR)/04_collect.done
fit: $(STATE_DIR)/05_fit.done
evaluate: $(STATE_DIR)/06_evaluate.done

$(STATE_DIR):
	@mkdir -p $@
	@mkdir -p $(MPLCONFIGDIR)

clean:
	@echo "Cleaning workspace..."
	@rm -rf $(WORKSPACE_PATH)
	@echo "Cleaned. Run 'make all' to restart."

$(STATE_DIR)/01_prepare.done: $(CONFIG_PATH) | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Stage 01] Prepare"
	@echo "========================================="
	mkdir -p "$(WORKSPACE_PATH)/commands" "$(WORKSPACE_PATH)/logs" "$(WORKSPACE_PATH)/models" \
		"$(WORKSPACE_PATH)/pairs" "$(WORKSPACE_PATH)/runs" "$(WORKSPACE_PATH)/state"
	$(PYTHON) $(MODELBRIDGE_APP) prepare --config "$(CONFIG_PATH)" --workspace "$(WORKSPACE_PATH)"
	touch $@

$(STATE_DIR)/02_hpo_train.done: $(STATE_DIR)/01_prepare.done | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Stage 02] HPO Train"
	@echo "========================================="
	RUNS_DIR="$(WORKSPACE_PATH)/runs/train"; \
	if [ ! -d "$$RUNS_DIR" ]; then \
		echo "Error: directory not found: $$RUNS_DIR"; \
		exit 1; \
	fi; \
	find "$$RUNS_DIR" -name "config.yaml" | LC_ALL=C sort | while IFS= read -r config_path; do \
		work_dir="$$(dirname "$$config_path")"; \
		echo "-----------------------------------------"; \
		echo "  [HPO] Starting optimization in: $$work_dir"; \
		echo "  [HPO] Config: $$config_path"; \
		aiaccel-hpo optimize --config "$$config_path"; \
	done
	touch $@

$(STATE_DIR)/03_hpo_test.done: $(STATE_DIR)/02_hpo_train.done | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Stage 03] HPO Test"
	@echo "========================================="
	RUNS_DIR="$(WORKSPACE_PATH)/runs/test"; \
	if [ ! -d "$$RUNS_DIR" ]; then \
		echo "Error: directory not found: $$RUNS_DIR"; \
		exit 1; \
	fi; \
	find "$$RUNS_DIR" -name "config.yaml" | LC_ALL=C sort | while IFS= read -r config_path; do \
		work_dir="$$(dirname "$$config_path")"; \
		echo "-----------------------------------------"; \
		echo "  [HPO] Starting optimization in: $$work_dir"; \
		echo "  [HPO] Config: $$config_path"; \
		aiaccel-hpo optimize --config "$$config_path"; \
	done
	touch $@

$(STATE_DIR)/04_collect.done: $(STATE_DIR)/03_hpo_test.done | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Stage 04] Collect Results"
	@echo "========================================="
	$(PYTHON) $(MODELBRIDGE_APP) collect --workspace "$(WORKSPACE_PATH)" --phase train
	$(PYTHON) $(MODELBRIDGE_APP) collect --workspace "$(WORKSPACE_PATH)" --phase test
	touch $@

$(STATE_DIR)/05_fit.done: $(STATE_DIR)/04_collect.done | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Stage 05] Fit Regression Model"
	@echo "========================================="
	$(PYTHON) $(MODELBRIDGE_APP) fit-model --workspace "$(WORKSPACE_PATH)"
	touch $@

$(STATE_DIR)/06_evaluate.done: $(STATE_DIR)/05_fit.done | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Stage 06] Evaluate Model"
	@echo "========================================="
	$(PYTHON) $(MODELBRIDGE_APP) evaluate --workspace "$(WORKSPACE_PATH)"
	touch $@
