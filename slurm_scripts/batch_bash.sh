#!/usr/bin/env bash

for file in ./scripts_uni_refute/*
do
  sbatch "$file"
  sleep 2
done
