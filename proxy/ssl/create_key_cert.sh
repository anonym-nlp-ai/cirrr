#!/usr/bin/env sh

# openssl req -batch -new -x509 -newkey rsa:4096 -nodes -sha256 \
#   -subj /CN=cir3.com/O=cir3 -days 3650 \
#   -keyout ./server.key \
#   -out ./server.crt


echo "Farkleberry" > pw_cert.txt
openssl req -x509 -newkey rsa:4096 -sha256 -passout file:pw_cert.txt\
  -subj "/C=UK/ST=GLN/L=LN/O=your_company country/OU=IT/CN=your_company.com" \
  -keyout ./server.key \
  -out ./server.crt  -days 365 \