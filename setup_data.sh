#!/usr/bin/env bash
# Downloads the catELMo binding-affinity dataset into ./data/
set -euo pipefail

mkdir -p data
cd data

if [ -f TCREpitopePairs.csv ]; then
  echo "TCREpitopePairs.csv already present, nothing to do."
  exit 0
fi

echo "Cloning catELMo (shallow)..."
rm -rf catELMo
git clone --depth 1 https://github.com/Lee-CBG/catELMo.git

echo "Extracting dataset..."
unzip -o -q catELMo/datasets/BindingAffinityPrediction.zip -d .
mv BindingAffinityPrediction/TCREpitopePairs.csv .

echo "Cleaning up..."
rm -rf catELMo BindingAffinityPrediction

echo
echo "Ready: data/TCREpitopePairs.csv"
wc -l TCREpitopePairs.csv
head -3 TCREpitopePairs.csv