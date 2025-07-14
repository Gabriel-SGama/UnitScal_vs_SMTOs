docker run --gpus all -e NVIDIA_VISIBLE_DEVICES=$1 --shm-size=16gb  -u 0 -it \
    -v `pwd`:/$USER/smto_analysis:rw \
    -v /home/lasi/Downloads/datasets/cityscapes/:/$USER/smto_analysis/datasets/city_ori \
    --hostname $HOSTNAME \
    --workdir /$USER/smto_analysis smto_analysis


# shared memory fix for cityscapes loading issue - https://github.com/pytorch/pytorch/issues/2244
# add: "--shm-size XG" to docker run (change X to the number of GB you want to allocate)
