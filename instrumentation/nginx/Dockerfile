ARG NGINX_VERSION=${NGINX_VERSION:-1.27.5}
FROM nginx:${NGINX_VERSION}

# Install apache2-utils to get htpasswd command
RUN apt-get update -y && apt-get install -y apache2-utils && rm -rf /var/lib/apt/lists/*

COPY ./conf.d/nginx.main.conf.template /etc/nginx/nginx.conf.template
COPY ./conf.d/nginx.conf.template /etc/nginx/conf.d/nginx.conf.template

COPY ./scripts/docker-entrypoint.sh /
RUN chmod +x ./docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]