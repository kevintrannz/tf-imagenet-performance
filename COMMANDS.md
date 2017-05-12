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

python tf_cnn_benchmarks.py \
    --local_parameter_device=cpu \
    --num_gpus=4 \
    --train_dir=${TRAIN_DIR} \
    --data_dir=${DATASET_DIR} \
    --variable_update=parameter_server \
    --num_batches=10000000000 \
    --summary_verbosity=1 \
    --save_summaries_steps=100 \
    --save_model_secs=600 \
    --batch_size=32 \
    --model=resnet50

