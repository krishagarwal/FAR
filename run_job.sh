#!/bin/bash
kubectl apply -f yaml/run_monarch.yaml
kubectl apply -f yaml/run_monarch_shared_LR.yaml
kubectl apply -f yaml/run_monarch_shared_local.yaml
kubectl apply -f yaml/run_monarch_max_constrain.yaml
kubectl apply -f yaml/run_monarch_max_constrain_reuse_qkv.yaml
