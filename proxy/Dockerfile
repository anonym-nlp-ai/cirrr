FROM nginx:1.23.2-alpine

COPY ./conf.d/default.conf.template /etc/nginx/conf.d/default.conf.template

COPY docker-entrypoint.sh /
RUN chmod +x ./docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]