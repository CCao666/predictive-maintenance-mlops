#!/bin/sh
set -eu

database_name="${MLFLOW_DB:-mlflow}"
database_exists="$(
    psql \
        --host "$POSTGRES_HOST" \
        --username "$POSTGRES_USER" \
        --dbname "$POSTGRES_DB" \
        --tuples-only \
        --no-align \
        --command "SELECT 1 FROM pg_database WHERE datname = '$database_name'"
)"

if [ "$database_exists" != "1" ]; then
    psql \
        --host "$POSTGRES_HOST" \
        --username "$POSTGRES_USER" \
        --dbname "$POSTGRES_DB" \
        --command "CREATE DATABASE \"$database_name\""
fi
