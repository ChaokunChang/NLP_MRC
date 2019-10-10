#!/bin/bash
mode=$1
key=$2
dataset="demo"
datapath="../data/$dataset"
train_data="$datapath/trainset/search.train.json"
dev_data="$datapath/devset/search.dev.json"
test_data="$datapath/testset/search.test.json"

vocab_path="../data/Svocabs/vocab_$dataset"_"$key"
model_path="../data/Smodels/model_$dataset"_"$key"
result_path="../data/Sresults/result_$dataset"_"$key"
summary_path="../data/Ssummaries/summary_$dataset"_"$key"
log_file="../logs/log_$dataset"_"$key"

if [ ! -d $vocab_path ]; then
  mkdir $vocab_path
fi

if [ ! -d $result_path ]; then
  mkdir $result_path
fi

if [ ! -d $model_path ]; then
  mkdir $model_path
fi

if [ ! -d $summary_path ]; then
  mkdir $summary_path
fi

if [ ! -f $log_file ]; then
  touch $log_file
fi

python run.py 	--$mode --gpu 7 --epochs 10 --learning_rate 0.001 \
		--train_files $train_data  --dev_files $dev_data  --test_files $test_data \
		--vocab_dir $vocab_path \
		--model_dir $model_path \
		--result_dir $result_path \
		--summary_dir $summary_path \
		--log_path $log_file
