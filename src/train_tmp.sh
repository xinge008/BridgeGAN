#!/usr/bin/env bash

python -u train_triple_resnet.py --config ../exps/unit/largebird_triple.yaml --log ../logs \
2>&1|tee ${LOG}
