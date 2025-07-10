#!/bin/bash




# Run docker compose
docker compose -f "$(dirname "$0")/../docker-compose.yml" up -d --build