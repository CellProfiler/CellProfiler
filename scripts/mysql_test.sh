#!/usr/bin/env bash

set -euo pipefail

ensure_var () {
  if [ ! -d "${CONDA_PREFIX}/var/mysql" ]; then
    mkdir -p "${CONDA_PREFIX}/var/mysql"
  fi
}

config_mysql_db () {
  ensure_var

  mysqld --initialize-insecure \
         --user="$(whoami)" \
         --datadir="${CONDA_PREFIX}/var/mysql"
}

destroy_mysql_db () {
  rm -rf "${CONDA_PREFIX}"/var/mysql/*
}

start_mysql_db () {
  mysqld --datadir="${CONDA_PREFIX}/var/mysql" \
         --socket="${CONDA_PREFIX}/var/mysql/mysql.sock" \
         --port=3306 \
         --bind-address=127.0.0.1 \
         --daemonize
}

enter_mysql_db () {
  mysql --socket="${CONDA_PREFIX}/var/mysql/mysql.sock" -u root
}

stop_mysql_db () {
  mysqladmin --socket="${CONDA_PREFIX}/var/mysql/mysql.sock" -u root shutdown
}

# one time only
cp_db_setup () {
  config_mysql_db

  start_mysql_db

  mysql --socket="${CONDA_PREFIX}/var/mysql/mysql.sock" -u root <<'EOF'
CREATE DATABASE cellprofiler_test;
CREATE USER 'cellprofiler'@'localhost' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON cellprofiler_test.* TO 'cellprofiler'@'localhost' WITH GRANT OPTION;
FLUSH PRIVILEGES;
EOF

  stop_mysql_db
}

cp_dbtest_start () {
  export LC_ALL="en_US.UTF-8"
  export CP_MYSQL_TEST_HOST="127.0.0.1"
  export CP_MYSQL_TEST_USER="cellprofiler"
  export CP_MYSQL_TEST_PASSWORD="password"
  start_mysql_db
}

cp_dbtest_end () {
  #unset LC_ALL
  unset CP_MYSQL_TEST_HOST
  unset CP_MYSQL_TEST_USER
  unset CP_MYSQL_TEST_PASSWORD
  stop_mysql_db
}

cp_dbtest() {
  cp_dbtest_start

  pytest "${PIXI_PROJECT_ROOT}/tests/frontend/modules/test_exporttodatabase.py"

  cp_dbtest_end
}

usage() {
  cat <<EOF
Usage:
  mysql_test.sh [OPTIONS] [COMMAND]
    Setup and run mysql for testing.

Options:
  --help | -h            Usage (this)

Higher Level Commands:
  config                 Configure and initialize mysql for cellprofiler database testing (one time)
  clear-config           Clear previous mysql database (WARNING: destroys all previous data)
  run                    Run cellprofiler database tests
  start                  Start the cellprofiler database
  stop                   Stop the cellprofiler database

Lower Level Commands:
  config-mysql           Configure blank mysql server
  destroy-mysq           Destroy mysql server data (WARNING: destroys all previous data)
  start-mysql            Start mysql server
  enter-mysql            Enter mysql instance (must have run --start-mysql first)
  stop-mysql             Stop mysql server (must have run --start-mysql first)
EOF
}
case "$1" in
    config)
      cp_db_setup
      exit $?
      ;;
    clear-config)
      cp_db_destroy
      exit $?
      ;;
    run)
      cp_dbtest
      exit $?
      ;;
    start)
      cp_dbtest_start
      exit $?
      ;;
    stop)
      cp_dbtest_end
      exit $?
      ;;
    config-mysql)
      config_mysql_db
      exit $?
      ;;
    destroy-mysql)
      destroy_mysql_db
      exit $?
      ;;
    start-mysql)
      start_mysql_db
      exit $?
      ;;
    enter-mysql)
      enter_mysql_db
      exit $?
      ;;
    stop-mysql)
      stop_mysql_db
      exit $?
      ;;
    -h)
      usage
      exit 0
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Invalid argument: $1"
      exit 1
      ;;
esac
