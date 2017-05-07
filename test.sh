#!/bin/bash

 LANG=en_us_8859_1
n=6
input_start=2013-10-01
input_end=$(date --date='-'$n' month' +'%Y-%m-%d')

startdate=$(date -I -d "$input_start") || exit -1
enddate=$(date -I -d "$input_end")     || exit -1

d="$startdate"
Dates=$(while [ "$(date -d "$d" +%Y%m%d)" -lt "$(date -d "$enddate" +%Y%m%d)" ]; do 
  echo $(date -d "$d" +%Y-%B)
  d=$(date -I -d "$d + 1 month")
	done)

aux=$(echo "$Dates")
echo "$aux"
