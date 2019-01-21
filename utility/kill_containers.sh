experiment_name=$1
docker ps --format '{{.Names}}' | grep "$experiment_name*" | awk '{print $1}' | xargs -I {} docker rm -f {}
