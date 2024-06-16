#!/bin/sh



export APP_MODULE=${APP_MODULE-app.main:app}
export HOST=${HOST:-0.0.0.0}
export PORT=${ASGIWS_PORT}

# run uvicorn
exec uvicorn --reload --host $HOST --port $PORT "$APP_MODULE"

# run gunicorn
# exec gunicorn --bind $HOST:$PORT "$APP_MODULE" -k uvicorn.workers.UvicornWorker