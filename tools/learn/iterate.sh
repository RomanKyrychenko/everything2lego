#!/bin/bash
# Iterate: generate negatives from current sampler, train network, repeat.
# Each round picks up the previous round's network for the next generation.
set -e
cd "$(dirname "$0")/../.."
ROUNDS=${1:-3}
for r in $(seq 1 $ROUNDS); do
    echo "=== round $r/$ROUNDS: negatives ==="
    .venv-arm/bin/python tools/learn/negatives.py --models 60 --iters 15000
    echo "=== round $r/$ROUNDS: train ==="
    .venv-arm/bin/python tools/learn/proposal_net.py train
done
