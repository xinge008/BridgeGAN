#!/usr/bin/env bash

python -u translate_triple_resnet.py --config ../exps/unit/eval_front2bird.yaml --log ../logs \
 --trans_alone 1 --a2b 1 --weights outputs/unit/_gen_00008000.pkl --output_folder outputs/result/8000/ \
2>&1|tee ${LOG}

