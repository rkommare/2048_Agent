docker build -t 2048_image .
docker run -it --mount type=bind,src=./container_files,dst=/agent/container_files --name 2048_container 2048_image