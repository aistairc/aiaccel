# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

$(train_path)/.train.done: | stage2 $(train_path)/config.yaml
	@set -e; \
	set -- \
		$(train_path)/train.* \
		$(train_path)/merged_config.yaml \
		$(train_path)/checkpoints \
		$(train_path)/events* \
		$(train_path)/hparams.yaml; \
	existing=""; \
	for file in "$$@"; do [ -e "$$file" ] && existing="$$existing $$file"; done; \
	if [ -n "$$existing" ]; then \
		echo "Found existing files/directories:"; printf '  %s\n' $$existing; \
		read -r -p "Delete these? [y/N]: " ans; \
		case "$$ans" in y|Y) rm -r $$existing ;; *) echo "Aborted."; exit 1 ;; esac; \
	fi
	$(cmd) train $(job_ops) \
		--n_gpus=$(shell aiaccel-config get-value $(train_path)/config.yaml n_gpus) \
		--walltime=$(shell aiaccel-config get-value $(train_path)/config.yaml walltime) \
		$(job_ops) $(train_path)/train.log -- \
			python -m aiaccel.torch.apps.train $(train_path)/config.yaml
	touch $@
