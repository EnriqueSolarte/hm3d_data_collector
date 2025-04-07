#!/bin/bash
docker build --build-arg USER=$(whoami) --build-arg USER_UID=$(id -u) -t hm3d_collector .
