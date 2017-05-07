#!/bin/bash


DIRECTORY="/home/patricio/Documents/Trial/Historys"
DIRECTORY2="/home/patricio/Documents/Trial/Nuevo"

if [ -d "$DIRECTORY" ] && [ -d "$DIRECTORY2" ];then
    if [ "$(ls -A $DIRECTORY)" ]; then
	echo "NON-EMPTY"
    else
	echo "EMPTY"
	fi
    else
	echo "FALTAN FOLDERS"
	fi
