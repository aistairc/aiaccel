# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

# general rules
STAGES = $(foreach num_of_stage,$(shell seq $(min_stage) $(max_stage)),stage$(num_of_stage))

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

$(STAGES):status .WAIT | $($@_dependencies)
	@echo "\033[1;34m$@ finished\033[0m"

$(STAGES:%=skip-%):
	echo $(patsubst skip-%,%,$@)_dependencies
	touch $($(patsubst skip-%,%,$@)_dependencies)
