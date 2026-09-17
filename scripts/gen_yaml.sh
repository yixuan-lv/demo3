#!/bin/bash
NAME=$1; RANK=$2; LR=$3; QBIT=$4; QMETHOD=$5; OUTDIR=$6; RUNNAME=$7
YAML=/root/demo3/exp/yaml/${NAME}.yaml

cp /root/demo3/train_qlora.yaml $YAML

sed -i "s|^lora_rank:.*|lora_rank: ${RANK}|" $YAML
sed -i "s|^learning_rate:.*|learning_rate: ${LR}|" $YAML
sed -i "s|^output_dir:.*|output_dir: ${OUTDIR}|" $YAML
sed -i "s|^swanlab_run_name:.*|swanlab_run_name: ${RUNNAME}|" $YAML

if [ "${QBIT}" = "none" ]; then
  sed -i '/^quantization_bit:/d' $YAML
  sed -i '/^quantization_method:/d' $YAML
  sed -i "s|^per_device_train_batch_size:.*|per_device_train_batch_size: 2|" $YAML
  sed -i "s|^gradient_accumulation_steps:.*|gradient_accumulation_steps: 8|" $YAML
else
  sed -i "s|^quantization_bit:.*|quantization_bit: ${QBIT}|" $YAML
  sed -i "s|^quantization_method:.*|quantization_method: ${QMETHOD}|" $YAML
fi

echo "[gen_yaml] $YAML"
grep -E "lora_rank|learning_rate|quantization|per_device_train_batch_size|gradient_accumulation_steps|output_dir|swanlab_run_name" $YAML
