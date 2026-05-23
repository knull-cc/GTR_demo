#!/usr/bin/env bash
set -e

bash scripts/crosslink/etth1.sh
bash scripts/crosslink/etth2.sh
bash scripts/crosslink/ettm1.sh
bash scripts/crosslink/ettm2.sh
bash scripts/crosslink/weather.sh
bash scripts/crosslink/electricity.sh
bash scripts/crosslink/traffic.sh
bash scripts/crosslink/solar.sh
# bash scripts/crosslink/pems03.sh
# bash scripts/crosslink/pems04.sh
# bash scripts/crosslink/pems07.sh
# bash scripts/crosslink/pems08.sh
