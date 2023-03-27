#!/usr/bin/bash
IMAGE_NAME=nsight
USER_ID=$(id -u)
GROUP_ID=$(id -g)

docker run --privileged --rm -it \
  --ipc host --gpus all --network host \
  -u $USER_ID:$GROUP_ID -w $(pwd) \
  -v /home/$USER:/home/$USER \
  $IMAGE_NAME fixuid /bin/bash