# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

mode ?= single

include $(mode).mk

.DEFAULT_GOAL := help

recipe_root ?= ./recipes
experiment := $(patsubst %/config.yaml,%,$(recipe))
recipe_path := $(recipe_root)/$(recipe)
train_path ?= ./runs/$(subst /,_,$(experiment))
data_root ?= $(shell aiaccel-config get-value $(recipe_path) datamodule.common_args.root)

cmd ?= aiaccel-job pbs
job_ops ?=

min_stage := 1
max_stage := 3
train_stage := 3

stage1_description := prepare the training directory
stage2_description := generate config.yaml with workflow metadata
stage3_description := launch training with aiaccel-job

stage1_dependencies := $(train_path)/.prepare.done
stage2_dependencies := $(train_path)/.config.done
stage3_dependencies := $(train_path)/.train.done

PRINT_VARIABLES := \
	recipe \
	recipe_path \
	train_path \
	data_root \
	cmd \
	job_ops

.PHONY: config train clean

$(train_path)/.prepare.done:
	mkdir -p $(train_path)
	python -c "from torchvision.datasets import CIFAR10; root=r'$(abspath $(data_root))'; CIFAR10(root=root, train=True, download=True); CIFAR10(root=root, train=False, download=True)"
	touch $@

$(train_path)/config.yaml: $(recipe_path) | $(train_path)/.prepare.done
	cp $(recipe_path) $@

$(train_path)/.config.done: $(train_path)/config.yaml
	touch $@

clean:
	rm -rf $(train_path)

include $(shell aiaccel-workflow template stages.mk)
include $(shell aiaccel-workflow template train.mk)
