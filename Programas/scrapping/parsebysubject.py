#!/usr/bin/env python

import json

with open ('./extropy_by_subject.json') as json_data:
	d=json.load(json_data)

for elements in d:
	name=elements["asunto"]+".txt2"
	content=elements["cuerpo"]
	name = name.replace("/","")
	file=open(name,"w")
	for contenido in content:
		file.write(contenido)
	file.close()
