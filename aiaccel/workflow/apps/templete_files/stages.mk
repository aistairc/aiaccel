# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

# general rules
STAGES = stage0 stage1 stage2 stage3 stage4 stage5

.NOTPARALLEL: check_train_artifacts 
.PHONY: all clean status scores check_train_artifacts $(STAGES) $(STAGES:%=skip-%) 
.WAIT:

# general rules
all: $(STAGES)

%: status

.stage%.done:
	touch $@

status:
	@echo ================================================================
	@$(foreach var,$(PRINT_VARIABLES),echo $(var): $($(var));)
	@echo ================================================================

stage1: status .WAIT | $(stage1_dependencies)
	@echo -e "\033[1;34mStage1 finished\033[0m"

skip-stage1:
	touch $(stage1_dependencies)
