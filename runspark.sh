#!/bin/bash

pip install pandas click numpy findspark ipython

export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/build:$PYTHONPATH

export PYSPARK_PYTHON=/usr/bin/python

PYSPARK_DRIVER_PYTHON=ipython

MASTER=spark://master:7077 ./bin/pyspark

IPYTHON=1 /usr/bin/pyspark

#PYSPARK_DRIVER_PYTHON=/usr/local/bin/ipython pyspark --master spark://master:7077

ipython spark.py
