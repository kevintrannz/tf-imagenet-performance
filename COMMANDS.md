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
# Inference benchmarking on Inception, MobileNets, GoogleNet, ....
# =========================================================================== #
# ~1100 images/sec on GTX Titan X
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/inception_v4.ckpt
python -u tf_cnn_benchmarks_slim.py \
    --eval=True \
    --local_parameter_device=cpu \
    --train_dir=${TRAIN_DIR} \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --model=inceptionv4 \
    --model_scope=v/InceptionV4 \
    --ckpt_scope=InceptionV4 \
    --variable_update=parameter_server \
    --summary_verbosity=1 \
    --num_gpus=1 \
    --num_batches=100 \
    --batch_size=32

# ~2100 images/sec on GTX Titan X
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/inception_v3.ckpt
python -u tf_cnn_benchmarks_slim.py \
    --eval=True \
    --local_parameter_device=cpu \
    --train_dir=${CHECKPOINT_PATH} \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --model=inceptionv3 \
    --model_scope=v/InceptionV3 \
    --ckpt_scope=InceptionV3 \
    --variable_update=parameter_server \
    --summary_verbosity=1 \
    --num_gpus=1 \
    --num_batches=100 \
    --batch_size=32

#  images/sec on GTX Titan X
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/inception_v2.ckpt
python -u tf_cnn_benchmarks_slim.py \
    --eval=True \
    --local_parameter_device=cpu \
    --train_dir=${CHECKPOINT_PATH} \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --data_format=NHWC \
    --model=inceptionv2 \
    --model_scope=v/InceptionV2 \
    --ckpt_scope=InceptionV2 \
    --variable_update=parameter_server \
    --summary_verbosity=1 \
    --num_gpus=1 \
    --num_batches=100 \
    --batch_size=32

# ~6700 images/sec on GTX Titan X
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/inception_v1.ckpt
python -u tf_cnn_benchmarks_slim.py \
    --eval=True \
    --local_parameter_device=cpu \
    --train_dir=${CHECKPOINT_PATH} \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --data_format=NHWC \
    --model=inceptionv1 \
    --model_scope=v/InceptionV1 \
    --ckpt_scope=InceptionV1 \
    --variable_update=parameter_server \
    --summary_verbosity=1 \
    --num_gpus=1 \
    --num_batches=100 \
    --batch_size=32

# ~7000 images/sec on GTX Titan X
CHECKPOINT_PATH=./checkpoints/mobilenets.ckpt
python -u tf_cnn_benchmarks_slim.py \
    --eval=True \
    --local_parameter_device=cpu \
    --train_dir=${CHECKPOINT_PATH} \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --model=mobilenets_caffe \
    --model_scope=v/MobileNets \
    --ckpt_scope=MobileNets \
    --variable_update=parameter_server \
    --summary_verbosity=1 \
    --num_gpus=1 \
    --num_batches=100 \
    --batch_size=32


# =========================================================================== #
# MobileNets training SLIM version
# =========================================================================== #
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
TRAIN_DIR=/media/paul/DataExt4/ImageNet/training/logs/mobilenet_002

CUDA_VISIBLE_DEVICES=0,1

DATASET_DIR=/media/imagenet/dataset
TRAIN_DIR=/media/imagenet/training/logs/mobilenet_003

    --pretrain_dir=${TRAIN_DIR} \

CUDA_VISIBLE_DEVICES=0,1 nohup python -u tf_cnn_benchmarks_slim.py \
    --local_parameter_device=cpu \
    --train_dir=${TRAIN_DIR} \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --model=mobilenets_caffe \
    --variable_update=parameter_server \
    --num_batches=1000000000000 \
    --summary_verbosity=1 \
    --save_summaries_steps=600 \
    --save_model_secs=1200 \
    --num_gpus=2 \
    --weight_decay=0.00001 \
    --learning_rate=0.1 \
    --learning_rate_decay_factor=0.94 \
    --num_epochs_per_decay=1.2 \
    --optimizer=rmsprop \
    --batch_size=128 &


CUDA_VISIBLE_DEVICES=2,3 nohup python -u tf_cnn_benchmarks_slim.py \
    --local_parameter_device=cpu \
    --train_dir=${TRAIN_DIR} \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --model=mobilenets_caffe \
    --variable_update=parameter_server \
    --num_batches=1000000000000 \
    --summary_verbosity=1 \
    --save_summaries_steps=600 \
    --save_model_secs=1200 \
    --num_gpus=2 \
    --weight_decay=0.000001 \
    --learning_rate=0.02 \
    --learning_rate_decay_factor=0.94 \
    --num_epochs_per_decay=2 \
    --optimizer=rmsprop \
    --batch_size=128 &



# =========================================================================== #
# MobileNets training
# =========================================================================== #
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
TRAIN_DIR=/media/paul/DataExt4/ImageNet/training/logs/mobilenet_002

CUDA_VISIBLE_DEVICES=0,1

DATASET_DIR=/media/imagenet/dataset
TRAIN_DIR=/media/imagenet/training/logs/mobilenet_003

    --pretrain_dir=${TRAIN_DIR} \

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

DATASET_DIR=/media/imagenet/dataset
TRAIN_DIR=/media/imagenet/training/logs/mobilenet_003

python -u tf_cnn_benchmarks.py \
    --eval=True \
    --local_parameter_device=cpu \
    --train_dir=${TRAIN_DIR} \
    --data_dir=${DATASET_DIR} \
    --resize_method=eval \
    --data_name=imagenet \
    --model=mobilenet \
    --variable_update=parameter_server \
    --summary_verbosity=1 \
    --num_gpus=4 \
    --num_batches=500 \
    --batch_size=25

# =========================================================================== #
# MobileNets Leaders training.
# =========================================================================== #
DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
TRAIN_DIR=/media/paul/DataExt4/ImageNet/training/logs/mobilenet_lead_001

DATASET_DIR=/media/imagenet/dataset
TRAIN_DIR=/media/imagenet/training/logs/mobilenet_lead_001

python -u tf_cnn_benchmarks_slim.py \
    --local_parameter_device=cpu \
    --train_dir=${TRAIN_DIR} \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --model=mobilenets_leaders \
    --variable_update=parameter_server \
    --num_batches=1000000000000 \
    --summary_verbosity=1 \
    --save_summaries_steps=600 \
    --save_model_secs=1200 \
    --num_gpus=1 \
    --weight_decay=0.00001 \
    --learning_rate=0.02 \
    --learning_rate_decay_factor=0.94 \
    --num_epochs_per_decay=1.0 \
    --optimizer=rmsprop \
    --batch_size=64


DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
CHECKPOINT_PATH=./checkpoints/mobilenets.ckpt

DATASET_DIR=/media/imagenet/dataset
python -u tf_cnn_benchmarks_slim.py \
    --eval=True \
    --local_parameter_device=cpu \
    --train_dir=${CHECKPOINT_PATH} \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --resize_method=crop_inception \
    --model=mobilenets_leaders \
    --variable_update=parameter_server \
    --summary_verbosity=1 \
    --num_gpus=4 \
    --num_batches=1000 \
    --batch_size=50



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


CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/inception_v3.ckpt
python -u tf_cnn_benchmarks_slim.py \
    --eval=True \
    --local_parameter_device=cpu \
    --train_dir=${CHECKPOINT_PATH} \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --model=inceptionv3 \
    --model_scope=v/InceptionV3 \
    --ckpt_scope=InceptionV3 \
    --variable_update=parameter_server \
    --summary_verbosity=1 \
    --num_gpus=1 \
    --num_batches=391 \
    --batch_size=32

DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
CHECKPOINT_PATH=./checkpoints/mobilenets.ckpt
python -u tf_cnn_benchmarks_slim.py \
    --eval=True \
    --local_parameter_device=cpu \
    --train_dir=${CHECKPOINT_PATH} \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --resize_method=eval \
    --model=mobilenets \
    --model_scope=v/MobileNets \
    --ckpt_scope=MobileNets \
    --variable_update=parameter_server \
    --summary_verbosity=1 \
    --num_gpus=1 \
    --num_batches=500 \
    --batch_size=100


DATASET_DIR=/media/paul/DataExt4/ImageNet/Dataset
CHECKPOINT_PATH=/media/paul/DataExt4/ImageNet/Training/ckpts/vgg_16.ckpt
python -u tf_cnn_benchmarks_slim.py \
    --eval=True \
    --local_parameter_device=cpu \
    --train_dir=${CHECKPOINT_PATH} \
    --data_dir=${DATASET_DIR} \
    --data_name=imagenet \
    --resize_method=crop \
    --model=vgg16 \
    --model_scope=v/vgg16 \
    --ckpt_scope=vgg_16 \
    --variable_update=parameter_server \
    --summary_verbosity=1 \
    --num_gpus=1 \
    --num_batches=50 \
    --batch_size=10
