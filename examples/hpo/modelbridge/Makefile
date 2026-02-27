SHELL := /bin/bash

# Directories
WORKSPACE_DIR ?= workspace
STATE_DIR := $(WORKSPACE_DIR)/state
LOGS_DIR := $(WORKSPACE_DIR)/logs
CONFIG_DIR := config
SCRIPTS_DIR := scripts

# Default configuration file
CONFIG_FILE ?= $(CONFIG_DIR)/config.yaml

# Shared runtime options
PYTHON ?= python
MPLCONFIGDIR ?= $(WORKSPACE_DIR)/.mplconfig

# Export variables for scripts
export CONFIG_FILE
export LOGS_DIR
export MPLCONFIGDIR
export PYTHON
export SCRIPTS_DIR
export STATE_DIR
export WORKSPACE_DIR

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
	@rm -rf $(WORKSPACE_DIR)
	@echo "Cleaned. Run 'make all' to restart."

$(STATE_DIR)/01_prepare.done: $(CONFIG_FILE) | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Stage 01] Prepare"
	@echo "========================================="
	bash $(SCRIPTS_DIR)/prepare.sh
	touch $@

$(STATE_DIR)/02_hpo_train.done: $(STATE_DIR)/01_prepare.done | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Stage 02] HPO Train"
	@echo "========================================="
	bash $(SCRIPTS_DIR)/run_hpo.sh train
	touch $@

$(STATE_DIR)/03_hpo_test.done: $(STATE_DIR)/02_hpo_train.done | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Stage 03] HPO Test"
	@echo "========================================="
	bash $(SCRIPTS_DIR)/run_hpo.sh test
	touch $@

$(STATE_DIR)/04_collect.done: $(STATE_DIR)/03_hpo_test.done | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Stage 04] Collect Results"
	@echo "========================================="
	bash $(SCRIPTS_DIR)/collect.sh
	touch $@

$(STATE_DIR)/05_fit.done: $(STATE_DIR)/04_collect.done | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Stage 05] Fit Regression Model"
	@echo "========================================="
	bash $(SCRIPTS_DIR)/fit.sh
	touch $@

$(STATE_DIR)/06_evaluate.done: $(STATE_DIR)/05_fit.done | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Stage 06] Evaluate Model"
	@echo "========================================="
	bash $(SCRIPTS_DIR)/evaluate.sh
	touch $@
