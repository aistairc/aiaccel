SHELL := /bin/bash

cmd := $(if $(JOB_COMMAND),$(JOB_COMMAND),aiaccel-job local)
job_ops ?=
PYTHON ?= python

MB_WORKDIR ?= work/modelbridge
MODELBRIDGE_CONFIG ?= $(MB_WORKDIR)/config.yaml
BRIDGE_OUTPUT_DIR ?= $(MB_WORKDIR)/outputs
MODELBRIDGE_LOG_DIR := $(MB_WORKDIR)/logs
AIACCEL_JOB_PROFILE ?= cpu

MODELBRIDGE_STAGES := mb-config mb-macro mb-micro mb-regress mb-summary

MB_CONFIG_SENTINEL := $(MB_WORKDIR)/.mb_config.done
MB_MACRO_SENTINEL := $(MB_WORKDIR)/.mb_macro.done
MB_MICRO_SENTINEL := $(MB_WORKDIR)/.mb_micro.done
MB_REGRESS_SENTINEL := $(MB_WORKDIR)/.mb_regress.done
MB_SUMMARY_SENTINEL := $(MB_WORKDIR)/.mb_summary.done

MODELBRIDGE_CLI := $(PYTHON) -m aiaccel.hpo.apps.modelbridge --config $(MODELBRIDGE_CONFIG)

PRINT_VARIABLES = \
	cmd job_ops -- \
	MODELBRIDGE_CONFIG \
	BRIDGE_OUTPUT_DIR \
	AIACCEL_JOB_PROFILE

.PHONY: modelbridge-run modelbridge-help clean-modelbridge status \
	$(MODELBRIDGE_STAGES) $(MODELBRIDGE_STAGES:%=skip-%)

modelbridge-run: $(MODELBRIDGE_STAGES)

status:
	@echo ================================================================
	@$(foreach var,$(PRINT_VARIABLES), \
	$(if $(filter --,$(var)), \
			echo "----------------------------------------------------------------";, \
			echo -e "\033[33m$(var)\033[0m:" '$($(var))';) \
	)
	@echo ================================================================

mb-config: status | $(MB_WORKDIR) $(MODELBRIDGE_LOG_DIR)
	$(cmd) $(AIACCEL_JOB_PROFILE) $(job_ops) $(MODELBRIDGE_LOG_DIR)/mb-config.log -- \
		$(MODELBRIDGE_CLI) --dry-run --print-config
	@touch $(MB_CONFIG_SENTINEL)

mb-macro: mb-config | $(MB_WORKDIR) $(MODELBRIDGE_LOG_DIR)
	$(cmd) $(AIACCEL_JOB_PROFILE) $(job_ops) $(MODELBRIDGE_LOG_DIR)/mb-macro.log -- \
		$(MODELBRIDGE_CLI) --phase macro
	@touch $(MB_MACRO_SENTINEL)

mb-micro: mb-macro | $(MB_WORKDIR) $(MODELBRIDGE_LOG_DIR)
	$(cmd) $(AIACCEL_JOB_PROFILE) $(job_ops) $(MODELBRIDGE_LOG_DIR)/mb-micro.log -- \
		$(MODELBRIDGE_CLI) --phase micro
	@touch $(MB_MICRO_SENTINEL)

mb-regress: mb-micro | $(MB_WORKDIR) $(MODELBRIDGE_LOG_DIR)
	$(cmd) $(AIACCEL_JOB_PROFILE) $(job_ops) $(MODELBRIDGE_LOG_DIR)/mb-regress.log -- \
		$(MODELBRIDGE_CLI) --phase regress
	@touch $(MB_REGRESS_SENTINEL)

mb-summary: mb-regress | $(MB_WORKDIR) $(MODELBRIDGE_LOG_DIR)
	$(cmd) $(AIACCEL_JOB_PROFILE) $(job_ops) $(MODELBRIDGE_LOG_DIR)/mb-summary.log -- \
		$(MODELBRIDGE_CLI) --phase summary
	@touch $(MB_SUMMARY_SENTINEL)

skip-mb-config:
	@mkdir -p $(MB_WORKDIR)
	@touch $(MB_CONFIG_SENTINEL)

skip-mb-macro:
	@mkdir -p $(MB_WORKDIR)
	@touch $(MB_MACRO_SENTINEL)

skip-mb-micro:
	@mkdir -p $(MB_WORKDIR)
	@touch $(MB_MICRO_SENTINEL)

skip-mb-regress:
	@mkdir -p $(MB_WORKDIR)
	@touch $(MB_REGRESS_SENTINEL)

skip-mb-summary:
	@mkdir -p $(MB_WORKDIR)
	@touch $(MB_SUMMARY_SENTINEL)

clean-modelbridge:
	@rm -rf $(BRIDGE_OUTPUT_DIR) $(MODELBRIDGE_LOG_DIR) \
		$(MB_CONFIG_SENTINEL) $(MB_MACRO_SENTINEL) $(MB_MICRO_SENTINEL) \
		$(MB_REGRESS_SENTINEL) $(MB_SUMMARY_SENTINEL)

modelbridge-help:
	@echo "Modelbridge targets"
	@echo "  modelbridge-run     Run $(MODELBRIDGE_STAGES) sequentially"
	@echo "  mb-<phase>         Run a single phase (config/macro/micro/regress/summary)"
	@echo "  skip-mb-<phase>    Mark a phase as completed by touching its sentinel"
	@echo "  clean-modelbridge  Remove logs, outputs, and sentinels"
	@echo ""
	@echo "Variables"
	@echo "  MODELBRIDGE_CONFIG=$(MODELBRIDGE_CONFIG)"
	@echo "  BRIDGE_OUTPUT_DIR=$(BRIDGE_OUTPUT_DIR)"
	@echo "  AIACCEL_JOB_PROFILE=$(AIACCEL_JOB_PROFILE)"
	@echo ""
	@echo "Example"
	@echo "  make modelbridge-run MODELBRIDGE_CONFIG=./config/modelbridge.yaml"

$(MB_WORKDIR):
	@mkdir -p $@

$(MODELBRIDGE_LOG_DIR): | $(MB_WORKDIR)
	@mkdir -p $@

.DEFAULT_GOAL := modelbridge-help
