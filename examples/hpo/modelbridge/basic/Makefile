SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

EXAMPLE_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PROJECT_ROOT := $(abspath $(EXAMPLE_DIR)/../../../..)
AIACCEL_HPO ?= aiaccel-hpo
AIACCEL_WORKFLOW ?= aiaccel-workflow
AIACCEL_MODELBRIDGE ?= aiaccel-modelbridge
MODELBRIDGE_CMD ?= $(AIACCEL_MODELBRIDGE)
cmd ?= aiaccel-job local
job_ops ?=
WORKFLOW_STAGES_MK := $(shell $(AIACCEL_WORKFLOW) template stages.mk)

resolve_path = $(if $(filter /%,$(1)),$(1),$(EXAMPLE_DIR)/$(1))
run_hpo_configs = RUNS_DIR="$(WORKSPACE_PATH)/runs/$(1)"; \
	if [ ! -d "$$RUNS_DIR" ]; then \
		echo "Error: directory not found: $$RUNS_DIR"; \
		exit 1; \
	fi; \
	find "$$RUNS_DIR" -name "config.yaml" | LC_ALL=C sort | while IFS= read -r config_path; do \
		work_dir="$$(dirname "$$config_path")"; \
		echo "-----------------------------------------"; \
		echo "  [HPO] Starting optimization in: $$work_dir"; \
		echo "  [HPO] Config: $$config_path"; \
		$(cmd) cpu $(job_ops) "$$work_dir/optimize.log" -- \
			$(AIACCEL_HPO) optimize --config "$$config_path"; \
	done

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

.DEFAULT_GOAL := all

min_stage := 1
max_stage := 6

PRINT_VARIABLES := \
	CONFIG_FILE \
	CONFIG_PATH \
	WORKSPACE_DIR \
	WORKSPACE_PATH \
	STATE_DIR \
	LOGS_DIR \
	AIACCEL_HPO \
	AIACCEL_MODELBRIDGE \
	AIACCEL_WORKFLOW \
	MODELBRIDGE_CMD \
	cmd \
	job_ops \
	PYTHON

stage1_description := prepare workspace and generated HPO configs
stage1_dependencies := $(STATE_DIR)/01_prepare.done

stage2_description := run train HPO studies
stage2_dependencies := $(STATE_DIR)/02_hpo_train.done

stage3_description := run test HPO studies
stage3_dependencies := $(STATE_DIR)/03_hpo_test.done

stage4_description := collect train and test HPO results
stage4_dependencies := $(STATE_DIR)/04_collect.done

stage5_description := fit the regression model
stage5_dependencies := $(STATE_DIR)/05_fit.done

stage6_description := evaluate the fitted model
stage6_dependencies := $(STATE_DIR)/06_evaluate.done

.PHONY: all prepare hpo-train hpo-test collect fit evaluate

all: evaluate
	@echo "========================================="
	@echo "  Modelbridge Pipeline Completed Successfully"
	@echo "========================================="

prepare: stage1
hpo-train: stage2
hpo-test: stage3
collect: stage4
fit: stage5
evaluate: stage6

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
	$(cmd) cpu $(job_ops) "$(LOGS_DIR)/01_prepare.log" -- \
		$(MODELBRIDGE_CMD) prepare --config "$(CONFIG_PATH)" --workspace "$(WORKSPACE_PATH)"
	touch $@

$(STATE_DIR)/02_hpo_train.done: $(STATE_DIR)/01_prepare.done | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Stage 02] HPO Train"
	@echo "========================================="
	$(call run_hpo_configs,train)
	touch $@

$(STATE_DIR)/03_hpo_test.done: $(STATE_DIR)/02_hpo_train.done | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Stage 03] HPO Test"
	@echo "========================================="
	$(call run_hpo_configs,test)
	touch $@

$(STATE_DIR)/04_collect.done: $(STATE_DIR)/03_hpo_test.done | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Stage 04] Collect Results"
	@echo "========================================="
	$(cmd) cpu $(job_ops) "$(LOGS_DIR)/04_collect_train.log" -- \
		$(MODELBRIDGE_CMD) collect --workspace "$(WORKSPACE_PATH)" --phase train
	$(cmd) cpu $(job_ops) "$(LOGS_DIR)/04_collect_test.log" -- \
		$(MODELBRIDGE_CMD) collect --workspace "$(WORKSPACE_PATH)" --phase test
	touch $@

$(STATE_DIR)/05_fit.done: $(STATE_DIR)/04_collect.done | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Stage 05] Fit Regression Model"
	@echo "========================================="
	$(cmd) cpu $(job_ops) "$(LOGS_DIR)/05_fit.log" -- \
		$(MODELBRIDGE_CMD) fit-model --workspace "$(WORKSPACE_PATH)"
	touch $@

$(STATE_DIR)/06_evaluate.done: $(STATE_DIR)/05_fit.done | $(STATE_DIR)
	@echo "========================================="
	@echo "  [Stage 06] Evaluate Model"
	@echo "========================================="
	$(cmd) cpu $(job_ops) "$(LOGS_DIR)/06_evaluate.log" -- \
		$(MODELBRIDGE_CMD) evaluate --workspace "$(WORKSPACE_PATH)"
	touch $@

include $(WORKFLOW_STAGES_MK)
