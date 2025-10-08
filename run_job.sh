#!/bin/bash
kubectl apply -f yaml/run_true_monarch_slow.yaml
kubectl apply -f yaml/run_true_monarch_fast.yaml
