#!/usr/bin/env bash
# Download the dev sample used by this repo (~96 MB total, public, no login).
set -euo pipefail
BASE=https://physionet.org/files/chbmit/1.0.0
mkdir -p data/raw/chb01 data/raw/chb02
cd data/raw/chb01
wget -nc $BASE/chb01/chb01-summary.txt
wget -nc $BASE/chb01/chb01_03.edf
wget -nc $BASE/chb01/chb01_03.edf.seizures
wget -nc $BASE/chb01/chb01_04.edf
wget -nc $BASE/chb01/chb01_04.edf.seizures
cd ../chb02
wget -nc $BASE/chb02/chb02-summary.txt
wget -nc $BASE/chb02/chb02_16.edf
wget -nc $BASE/chb02/chb02_16.edf.seizures
