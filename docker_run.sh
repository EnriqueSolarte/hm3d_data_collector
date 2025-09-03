 #!/bin/bash
xhost +local:
CONTAINER_NAME="hm3d-data-collector-dev"

# this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' is already running. Attaching..."
    docker exec -it "${CONTAINER_NAME}" bash
else
    echo "Starting new container '${CONTAINER_NAME}'..."
    docker run -itd \
        --rm \
        --gpus all --runtime=nvidia \
        --name "${CONTAINER_NAME}" \
        --privileged \
        --network=host --ipc=host \
        --env="DISPLAY=${DISPLAY}" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        -v /media/datasets/habitat/:/media/datasets/habitat/ \
        -v /media/datasets/hm_semantic_maps/map_free_navigation:/data_collected \
        -v /media/q200/kike/docker_ws/test_bases/tests:/tests \
        habitat_data_collector:dev
fi
