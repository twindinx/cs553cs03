#!/bin/sh

set -e

prometheus-node-exporter --web.listen-address=":9100" --no-collector.systemd &

exec python main.py