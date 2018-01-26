#!/bin/bash
python train.py --ds-name 'CUB2011'\
                --ds-path '/mnt/45B9E78077FAE8C2/Dev Files/Datasets/CUB2011'\
                --save-dir 'results-lr10'\
                --batch-size 30\
                --num-workers 3\
                --save-every 5\
                --epochs 1

