#!/usr/bin/env python

import email
import re
import sys,os
from bs4 import BeautifulSoup
for x in range(2003, 2018):
	infile = str(x)	
	content = ""
	listamails = ""
	listacorreos = ""
	mail = ""
	print(infile)
	email_json = "["
	with open(infile,"r") as f:
		content = f.read()
		listamails = content
		f.close()
	i=1
	verifica = "test"
	listacorreos = re.split('From ', listamails)
	while (i<=len(listacorreos)-1):
		mail = listacorreos[i]
		mail = "From " + mail
		#mail = BeautifulSoup(mail).text
		b = email.message_from_string(mail)
		from_val = b['From']
		if(type(from_val) == type(verifica)):
			from_val = re.sub(r'\(.*\)','',from_val)
			from_val = from_val.replace(" at ", "@")
			from_val = from_val.replace(" ","")
		email_json = email_json + "{\"From\"" + ":" + "\"" + str(from_val) + "\"" + ","
		date_val = b['Date']
		email_json = email_json + "\"" + "Date" + "\"" + ":" + "\"" + str(date_val) + "\"" + "," 
		subject_val = b['Subject']
		if(type(subject_val) == type(verifica)):
			subject_val = re.sub(r'\[.*\]','',subject_val)
			subject_val = subject_val.replace(" ","")
			subject_val = subject_val.replace("\"","")
		email_json = email_json + "\"" + "Subject" + "\"" + ":" + "\"" + str(subject_val) + "\"" + ","
		reply_val = b['In-Reply-To']
		if (type(reply_val) == type(verifica)):
			reply_val = reply_val.replace("<","")
			reply_val = reply_val.replace(">","")
			reply_val = reply_val.replace(" ","")
			reply_val = reply_val.replace("\"","")
		email_json = email_json + "\"" + "In-Reply-To" + "\"" + ":" + "\"" + str(reply_val) + "\"" + ","
		reference_val = b['References']
		if(type(reference_val) == type(verifica)):
			reference_val = reference_val.replace("\n","")
			reference_val = reference_val.replace("> <",",")
			reference_val = reference_val.replace("<","[")
			reference_val = reference_val.replace(">","]")
		email_json = email_json + "\"" + "References" + "\"" + ":" + "\"" + str(reference_val) + "\"" + ","
		messageId_val = b['Message-ID']
		if(type(messageId_val) == type(verifica)):
			messageId_val = messageId_val.replace("<","")
			messageId_val = messageId_val.replace(">","")
		email_json = email_json + "\"" + "Message-ID" + "\"" + ":" + "\"" + str(messageId_val) + "\"" + ","
		body = ""
		if b.is_multipart():
			for part in b.walk():
					ctype = part.get_content_type()
					cdispo = str(part.get('Content-Disposition'))
					# skip any text/plain (txt) attachments
					if ctype == 'text/plain' and 'attachment' not in cdispo:
						#body = part.get_payload(decode=True)  # decode
						break
		# not multipart - i.e. plain text, no attachments, keeping fingers crossed
		else:
			body = b.get_payload(decode=True)
		body = str(body)
		body = body[1:]
		body = body.replace("\n","")
		body = body.replace("\"","")
		body = body.replace("'","")
		if(i==len(listacorreos)-1):
			email_json = email_json + "\"" + "Body" + "\"" + ":" + "\""+ body + "\"" + "}]"
		else:
			email_json = email_json + "\"" + "Body" + "\"" + ":" + "\""+ body + "\"" + "},"
		i = i + 1
		
	#print(email_json)
	outfile = "json_raw_"+str(x)
	print(outfile)
	with open(outfile,"w") as output:
		output.write(email_json)
		output.close()


