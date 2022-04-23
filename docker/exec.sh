# Run fish shell in the docker container.

. docker/init.sh
docker exec \
  -it \
  $CONTAINER_NAME bash 
