#!/bin/bash

#MACHINE_DRIVER= amazonec2
#AWS_ACCESS_KEY_ID= AKIAJZ2NYJKMMWARZ3PA
#AWS_SECRET_ACCESS_KEY= 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
#AWS_DEFAULT_REGION= us-west-2
# export AWS_INSTANCE_TYE=m3.large
for N in $(seq 4 4); do
    sudo docker-machine create --driver amazonec2 --amazonec2-access-key AKIAJZ2NYJKMMWARZ3PA --amazonec2-secret-key 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 --amazonec2-region us-west-2 dpa-node$N
sudo docker-machine ssh dpa-node$N sudo usermod -aG docker ubuntu
done


#7946
#docker exec
