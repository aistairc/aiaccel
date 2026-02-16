#!/bin/bash

LOG_FILE=$1
shift
aiaccel-hpo modelbridge run "$@" > "$LOG_FILE" 2>&1
