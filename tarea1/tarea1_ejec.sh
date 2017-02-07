#!/bin/bash
COUNTER=1
while [  $COUNTER -lt 31 ]; do
   if [ $COUNTER -lt 10 ]; then
    echo wget http://data.gdeltproject.org/events/2016120$COUNTER.export.CSV.zip >> descargas
    echo wget http://data.gdeltproject.org/events/2017010$COUNTER.export.CSV.zip >> descargas
   else
    echo wget http://data.gdeltproject.org/events/201612$COUNTER.export.CSV.zip >> descargas
    echo wget http://data.gdeltproject.org/events/201701$COUNTER.export.CSV.zip >> descargas
   fi
   let COUNTER=COUNTER+1
done
parallel -j0 -a descargas
echo PESO TOTAL ARCHIVOS $(du -h)
echo NUMERO TOTAL DE ARCHIVOS $(ls *.zip | wc -l)
wget http://gdeltproject.org/data/lookups/CSV.header.dailyupdates.txt
ls *.zip > archivos
mkdir Mexico
cat CSV.header.dailyupdates.txt > Mexico/headers
while IFS= read -r var
do
echo "zcat "$var"|awk -v OFS='\t' '{if(\$37 == \"MX\"||\$44 ==\"MX\") print}' >> Mexico/"$var".csv" >> procesa_csv
done < "archivos"
parallel -j0 -a procesa_csv
ls Mexico/*.csv > Mexico/carga
cat Mexico/headers > Mexico/Mexico.csv
while IFS= read -r var
do
echo "cat "$var" >> Mexico/Mexico.csv" >> Mexico/procesa_mexico
done < "Mexico/carga"
parallel -j0 -a Mexico/procesa_mexico
csvsql --db sqlite:///gdelt.db --insert Mexico/Mexico.csv
echo "fecha,num_noticias,goldstein"> Mexico/mexico_ts.csv
awk -F "\t" 'NR>1 {a[$2]+=$31;}{b[$2]++;}END{for (i in a)print i,",",b[i],",", a[i]/b[i];}' Mexico/Mexico.csv|sed 's/ /$
csvsql --db sqlite:///gdelt.db --insert Mexico/mexico_ts.csv
rm *.zip
cd Mexico/
rm *.csv

