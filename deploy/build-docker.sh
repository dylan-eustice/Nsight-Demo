#!/usr/bin/bash
docker build --network host -t nsight -f $(dirname "$0")/Dockerfile .