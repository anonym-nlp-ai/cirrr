#!/bin/sh


set -eu

envsubst '${AIMW_HOST} ${AIMW_PORT} ${NGINX_PORT} ${SSL_CERT_NAME} ${SSL_CERT_KEY_NAME} ${SSL_PASS_FILE} ${GRAFANA_PORT}' < /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf

exec "$@"