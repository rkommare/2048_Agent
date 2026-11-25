docker build -t 2048_image .
docker run -it --mount type=bind,src=.,dst=/agent --name 2048_container 2048_image