#!/bin/sh


set -eu

# Process the main nginx.conf template
envsubst < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf

# Process the stream configuration template
envsubst '${PROMETHEUS_SERVICE_HOST} ${PROMETHEUS_SERVICE_PORT} ${PROMETHEUS_PORT} ${SSL_CERT_NAME} ${SSL_CERT_KEY_NAME}' < /etc/nginx/conf.d/nginx.conf.template > /etc/nginx/conf.d/default.conf

exec "$@"