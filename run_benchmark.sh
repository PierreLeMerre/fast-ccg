#!/bin/bash
source julia/python/.venv/bin/activate
cd julia/benchmark
julia "$@"