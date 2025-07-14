#!/bin/bash
set -e

echo "📦 Creating buckets..."

awslocal s3 mb s3://personality-classifier-model-bucket
awslocal s3 mb s3://ml-pipeline-dvc-cache

echo "✅ Buckets created"