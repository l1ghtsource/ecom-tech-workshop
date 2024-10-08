#!/bin/bash

mkdir -p ecom-tech-workshop/src/output/checkpoint-1036
mkdir -p ecom-tech-workshop/src/output/checkpoint-562

huggingface-cli download lightsource/gemma2_9b_multilabel_lora_adapter_ver1 --revision main --cache-dir . --local-dir ecom-tech-workshop/src/output/checkpoint-1036
huggingface-cli download lightsource/gemma2_9b_multilabel_lora_adapter_ver2 --revision main --cache-dir . --local-dir ecom-tech-workshop/src/output/checkpoint-562

echo "Веса успешно скачаны!"