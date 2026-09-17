#!/bin/bash
NAME=$1; RANK=$2; LR=$3; QBIT=$4; QMETHOD=$5

EXP_ROOT=/root/demo3/exp
YAML=$EXP_ROOT/yaml/${NAME}.yaml
OUTDIR=/root/autodl-tmp/demo3/exp/${NAME}
PREDDIR=$EXP_ROOT/predict/${NAME}
LOGDIR=$EXP_ROOT/logs
RES=$EXP_ROOT/results/results.csv
GOLD=/root/demo3/data/bc2gm_test.json
RUNNAME=exp_${NAME}

mkdir -p $OUTDIR $PREDDIR $LOGDIR

if [ -f "$RES" ] && grep -q "^${NAME}," "$RES"; then
  echo "[skip] $NAME 已在 results.csv 中"
  exit 0
fi

echo "===== [$(date '+%F %T')] 开始 $NAME ====="

bash $EXP_ROOT/gen_yaml.sh $NAME $RANK $LR $QBIT $QMETHOD $OUTDIR $RUNNAME

cd /root/LLaMA-Factory
echo "--- 训练 $NAME ---"
llamafactory-cli train $YAML 2>&1 | tee $LOGDIR/${NAME}_train.log

cat > $LOGDIR/${NAME}_predict.yaml << YEOF
model_name_or_path: /root/models/Qwen2.5-7B-Instruct
adapter_name_or_path: ${OUTDIR}
template: qwen
finetuning_type: lora
eval_dataset: bc2gm_test
cutoff_len: 1024
max_new_tokens: 512
do_sample: false
output_dir: ${PREDDIR}
overwrite_output_dir: true
per_device_eval_batch_size: 8
predict_with_generate: true
do_predict: true
YEOF

echo "--- 推理 $NAME ---"
llamafactory-cli train $LOGDIR/${NAME}_predict.yaml 2>&1 | tee $LOGDIR/${NAME}_predict.log

echo "--- 评测 $NAME ---"
METRIC=$(python $EXP_ROOT/eval_one.py ${PREDDIR}/generated_predictions.jsonl $GOLD)
echo "${NAME},${RANK},${LR},${QBIT},${METRIC}" >> $RES
echo "===== $NAME 完成: P,R,F1 = $METRIC ====="
