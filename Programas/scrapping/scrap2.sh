#!/bin/bash

file="descargas"
if [ -f $file ]; then
rm $file
fi
for i in $(seq 2003 2017);do
    for j in January February March April May June July August September October November December;do
echo "curl http://lists.extropy.org/pipermail/extropy-chat/"$i-$j".txt >>"$i-$j".txt">> descargas
done
done

parallel -j0 -a descargas
find . -size +1000c -a  -size -2100c -delete
for i in $(seq 2003 2017); do
    pr -F $i*.txt > $i
    rm $i*.txt
done
for i in $(seq 2003 2017); do
sed 's/\xA0//g' $i > $i.l
mv $i.l $i
done
for i in $(seq 2003 2017); do
sed 's/\xED//g' $i > $i.l
mv $i.l $i
done
./txt2json.py
for i in $(seq 2003 2017); do
sed -e 's/\\//g'  json_raw_$i > $i.json
rm json_raw_$i
cat $i.json | jq -r '.'
done
file2="extropy.json"
if [ -f $file2 ]; then
rm $file2
fi
head -c -1 2003.json > extropy.json
for i in $(seq 2004 2017); do
echo "," >> extropy.json
sed '1s/^.//' 2004.json > extropy_raw
head -c -1 extropy_raw >> extropy.json
done
echo "]" >> extropy.json
rm extropy_raw
cat extropy.json | jq -r '.'
for i in $(seq 2003 2017); do
rm $i
rm $i.json
done
jq -r '[.[]|{"asunto":.Subject,"cuerpo":.Body}]|group_by(.asunto)|map({"asunto": .[0].asunto, "cuerpo": map(.cuerpo)})' extropy.json > extropy_by_subject.json
