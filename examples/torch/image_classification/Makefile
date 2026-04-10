# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

available_modes := single ddp
mode ?= single

include ./recipes/$(mode).mk

ifeq ($(filter $(mode),$(available_modes)),)
$(error Unsupported mode '$(mode)'. Choose one of: $(available_modes))
endif

.DEFAULT_GOAL := help

job_config_path ?= ./recipes/job_config.yaml
recipe_path := $(train_path)/config.yaml
data_root ?= $(shell aiaccel-config get-value $(train_path)/config.yaml datamodule.common_args.root)

cmd ?= aiaccel-job pbs --config $(job_config_path)
job_ops ?=

min_stage := 1
max_stage := 2
train_dependencies = stage1

.PHONY: train clean

PRINT_VARIABLES := \
	mode \
	recipe_path \
	train_path \
	data_root \
	cmd \
	job_ops

stage1_description := download CIFAR-10 to the configured data_root
stage1_dependencies := $(train_path)/.prepare.done

$(train_path)/.prepare.done:
	python ./recipes/scripts/prepare_dataset.py $(abspath $(data_root))
	touch $@

include $(shell aiaccel-workflow template stages.mk)
include $(shell aiaccel-workflow template train.mk)

stage2_description := launch aiaccel-job train with the selected config
stage2_dependencies := $(train_path)/.train.done

train: stage2

clean:
	rm $(train_path)/.*.done
