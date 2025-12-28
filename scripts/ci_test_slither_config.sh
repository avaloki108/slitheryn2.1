#!/usr/bin/env bash

### Test

if ! slitheryn "tests/*.json" --config "tests/config/slitheryn.config.json"; then
    echo "Config failed"
    exit 1
fi

