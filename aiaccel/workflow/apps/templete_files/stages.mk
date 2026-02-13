# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

# constant values
STAGES = $(foreach num_of_stage,$(shell seq $(min_stage) $(max_stage)),stage$(num_of_stage))

.NOTPARALLEL: check_train_artifacts 
.PHONY: all clean status scores check_train_artifacts $(STAGES) $(STAGES:%=skip-%)
.WAIT:

# general rules
all: $(STAGES)

%: status

status:
	@echo ================================================================
	@$(foreach var,$(PRINT_VARIABLES),echo $(var): $($(var));)
	@echo ================================================================

$(eval stage$(min_stage): status .WAIT $(stage$(min_stage)_dependencies))
$(foreach n,$(shell seq $$(($(min_stage)+1)) $(max_stage)), \
  $(eval stage$(n): status .WAIT $(stage$(n)_dependencies) stage$(shell echo $$(($(n)-1)))) \
)

$(STAGES):
	@echo "\033[1;34m$@ finished\033[0m"

$(STAGES:%=skip-%):
	touch $($(patsubst skip-%,%,$@)_dependencies)
