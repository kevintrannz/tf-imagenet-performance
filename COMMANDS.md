python tf_cnn_benchmarks.py \
    --local_parameter_device=cpu \
    --num_gpus=4 \
    --batch_size=32 \
    --model=vgg16 \
    --data_dir=/media/imagenet/dataset \
    --variable_update=parameter_server
