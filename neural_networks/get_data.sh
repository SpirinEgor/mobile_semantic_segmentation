#!/bin/bash

if [[ -z $1 ]]; then
    WORK_PATH "data"
else
    WORK_PATH=$1
    if [[ "${WORK_PATH: -1}" == "/" ]]; then
        WORK_PATH=${WORK_PATH::-1}
    fi
fi
echo "Download to $WORK_PATH/"

wget -c http://images.cocodataset.org/zips/train2017.zip -O $WORK_PATH/train2017.zip
unzip $WORK_PATH/train2017.zip -d $WORK_PATH
wget -c http://images.cocodataset.org/zips/val2017.zip -O $WORK_PATH/val2017.zip
unzip $WORK_PATH/val2017.zip -d $WORK_PATH
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip -O $WORK_PATH/annotations.zip
unzip $WORK_PATH/annotations.zip -d $WORK_PATH

if [[ $WORK_PATH != "data" ]]; then
    ln -s $WORK_PATH "data"
fi
