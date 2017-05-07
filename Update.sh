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

auxdates=$(echo "$Dates")

update_fun () {
DIRECTORY="/home/patricio/Documents/Trial/History"
DIRECTORY2="/home/patricio/Documents/Trial/Nuevo"
n=6
 rm $DIRECTORY2/*.txt
     LANG=en_us_8859_1
     for i in $(seq 0 $n);do
    dat=$( date --date='-'$i' month' +'%Y')'-'$( date --date='-'$i' month' +'%B' )
 curl http://lists.extropy.org/pipermail/extropy-chat/${dat}/date.html >>$DIRECTORY2/${dat}.txt
done

replacement=$( date --date='-'$n' month' +'%Y')'-'$( date --date='-'$n' month' +'%B' )
cp -T $DIRECTORY2/${replacement}.txt $DIRECTORY/${replacement}.txt
}

DIRECTORY="/home/patricio/Documents/Trial/History"
DIRECTORY2="/home/patricio/Documents/Trial/Nuevo"
DIRECTORY3="/home/patricio/Documents/Trial"

# init
# look for empty dir
if [ -d "$DIRECTORY" ] && [ -d "$DIRECTORY2" ]; then
	if [ "$(ls -A $DIRECTORY)" ]; then
	    update_fun
	else
	    for i in $auxdates;do
		curl http://lists.extropy.org/pipermail/extropy-chat/$i/date.html >>$DIRECTORY/$i.txt
	    done
	    update_fun
	fi
else
    rm -rf $DIRECTORY3
    mkdir $DIRECTORY3
    mkdir $DIRECTORY
    mkdir $DIRECTORY2
    for i in $auxdates;do
	curl http://lists.extropy.org/pipermail/extropy-chat/$i/date.html >>$DIRECTORY/$i.txt
    done
    update_fun
    fi
