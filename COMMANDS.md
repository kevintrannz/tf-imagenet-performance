# =========================================================================== #
# Benchmark scripts
# =========================================================================== #
DATASET_DIR=/media/imagenet/dataset
TRAIN_DIR=./logs/ssd_300_vgg_1

python tf_cnn_benchmarks.py \
    --local_parameter_device=cpu \
    --num_gpus=4 \
    --batch_size=32 \
    --model=vgg16 \
    --data_dir=/media/imagenet/dataset \
    --variable_update=parameter_server

DATASET_DIR=/media/imagenet/dataset
TRAIN_DIR=/media/imagenet/training/logs/resnet_001

DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
TRAIN_DIR=/media/paul/DataExt4/ImageNet/training/logs/resnet_001

python tf_cnn_benchmarks.py \
    --local_parameter_device=cpu \
    --train_dir=${TRAIN_DIR} \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --variable_update=parameter_server \
    --num_batches=10000000000 \
    --summary_verbosity=1 \
    --save_summaries_steps=100 \
    --save_model_secs=600 \
    --num_gpus=1 \
    --weight_decay=0.00004 \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.94 \
    --optimizer=rmsprop \
    --batch_size=32 \
    --model=resnet50


# =========================================================================== #
# MobileNets training
# =========================================================================== #
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
TRAIN_DIR=/media/paul/DataExt4/ImageNet/training/logs/mobilenet_002

CUDA_VISIBLE_DEVICES=0,1

DATASET_DIR=/media/imagenet/dataset
TRAIN_DIR=/media/imagenet/training/logs/mobilenet_003

CUDA_VISIBLE_DEVICES=0,1 nohup python -u tf_cnn_benchmarks.py \
    --local_parameter_device=cpu \
    --train_dir=${TRAIN_DIR} \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --model=mobilenet \
    --variable_update=parameter_server \
    --num_batches=1000000000000 \
    --summary_verbosity=1 \
    --save_summaries_steps=600 \
    --save_model_secs=1200 \
    --num_gpus=2 \
    --weight_decay=0.00001 \
    --learning_rate=0.1 \
    --learning_rate_decay_factor=0.94 \
    --num_epochs_per_decay=1.0 \
    --optimizer=rmsprop \
    --batch_size=128 &


CUDA_VISIBLE_DEVICES=0,1 nohup python -u tf_cnn_benchmarks.py \
    --local_parameter_device=cpu \
    --train_dir=${TRAIN_DIR} \
    --data_dir=${DATASET_DIR} \
    --pretrain_dir=${TRAIN_DIR} \
    --data_name=imagenet \
    --model=mobilenet \
    --variable_update=parameter_server \
    --num_batches=1000000000000 \
    --summary_verbosity=1 \
    --save_summaries_steps=1000 \
    --save_model_secs=1800 \
    --num_gpus=2 \
    --weight_decay=0.0001 \
    --learning_rate=0.1 \
    --learning_rate_decay_factor=0.94 \
    --num_epochs_per_decay=2.0 \
    --optimizer=rmsprop \
    --batch_size=64 &



nohup python -u tf_cnn_benchmarks.py \
    --local_parameter_device=cpu \
    --train_dir=${TRAIN_DIR} \
    --pretrain_dir=${TRAIN_DIR} \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --model=mobilenet \
    --variable_update=parameter_server \
    --num_batches=1000000000000 \
    --summary_verbosity=1 \
    --save_summaries_steps=1000 \
    --save_model_secs=1800 \
    --num_gpus=4 \
    --weight_decay=0.00001 \
    --learning_rate=0.01 \
    --learning_rate_decay_factor=0.94 \
    --num_epochs_per_decay=2.0 \
    --optimizer=rmsprop \
    --batch_size=32 &

python -u tf_cnn_benchmarks.py \
    --eval=True \
    --local_parameter_device=cpu \
    --train_dir=${TRAIN_DIR} \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --model=mobilenet \
    --variable_update=parameter_server \
    --summary_verbosity=1 \
    --num_gpus=4 \
    --num_batches=391 \
    --batch_size=32

# =========================================================================== #
# Benchmark Original vs Slim
# =========================================================================== #
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
DATASET_DIR=/media/imagenet/dataset

python -u tf_cnn_benchmarks.py \
    --local_parameter_device=cpu \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --model=vgg16 \
    --variable_update=parameter_server \
    --num_gpus=1 \
    --weight_decay=0.00001 \
    --learning_rate=0.01 \
    --learning_rate_decay_factor=0.94 \
    --num_epochs_per_decay=2.0 \
    --optimizer=rmsprop \
    --batch_size=32


python -u tf_cnn_benchmarks_slim.py \
    --local_parameter_device=cpu \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --model=vgg16 \
    --variable_update=parameter_server \
    --num_gpus=1 \
    --weight_decay=0.00001 \
    --learning_rate=0.01 \
    --learning_rate_decay_factor=0.94 \
    --num_epochs_per_decay=2.0 \
    --optimizer=rmsprop \
    --batch_size=32
