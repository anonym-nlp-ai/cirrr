FROM python:3.11 as requirements-stage

WORKDIR /tmp

# Install Poetry in Docker requirements-stage.
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

COPY ./pyproject.toml ./poetry.lock* /tmp/

# Allow installing dev dependencies to run tests
ARG INSTALL_DEV_ARG=false

RUN bash -c "if [ $INSTALL_DEV_ARG == true ] ; then poetry export -f requirements.txt --output requirements.txt --without-hashes --dev; \
 else poetry export -f requirements.txt --output requirements.txt --without-hashes ; fi"

# Final stage, the rest will be preserved in the final image.
FROM python:3.11

ARG UID
ARG GID
# use the value to set the ENV var default (run time)
ENV UID $UID
ENV GID $GID

RUN addgroup --system --gid $GID ai && adduser --system -u $UID --gid $GID ai

ENV HOME=/home/ai
ENV APP_HOME=/home/ai/aimw
WORKDIR $HOME
RUN mkdir aimw
RUN mkdir logs

ENV PYTHONDONTWRITEBYTECODE 1
# Py logs > container log in real time (without being first buffered).
ENV PYTHONUNBUFFERED 1

COPY --from=requirements-stage /tmp/requirements.txt $HOME/requirements.txt
RUN pip install --no-cache-dir --upgrade -r $HOME/requirements.txt

COPY ./app $APP_HOME/app
COPY ./conf $HOME/conf

# Copy run.sh
COPY run.sh $APP_HOME
RUN chmod +x $APP_HOME/run.sh

# chown all the files to the ai user
RUN echo $HOME
RUN chown -R $UID:$GID $HOME

WORKDIR $APP_HOME

# use port 8001:8001: `docker run -d -p 8001:8001 ai`. Check run.sh for more details
CMD ["./run.sh"]

# CMD uvicorn --reload --host $HOST --port $PORT app.main:app
# CMD gunicorn --bind 0.0.0.0:$PORT app.main:app -k uvicorn.workers.UvicornWorker 4