#!/bin/bash
# training and output the models
../../xgboost mushroom.conf
# output prediction task=pred 
../../xgboost mushroom.conf task=pred model_in=0002.model
# print the boosters of 00002.model in dump.raw.txt
../../xgboost mushroom.conf task=dump model_in=0002.model name_dump=dump.raw.txt 
# use the feature map in printing for better visualization
#../../xgboost mushroom.conf task=dump model_in=0002.model fmap=featmap.txt name_dump=dump.nice.txt
#cat dump.nice.txt

