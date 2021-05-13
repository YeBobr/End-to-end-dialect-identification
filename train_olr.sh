#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=1        #to-do-set
stop_stage=100 #to-do-set
ngpu=4         # number of gpus ("0" uses cpu, otherwise use gpu) #to-do-set
debugmode=1
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot 
log=100
whichformer=conformer

preprocess_config=conf/espnet_specaug.yaml
train_config=conf/train_olr_${whichformer}.yaml 

# others
accum_grad=2
n_iter_processes=32
lsm_weight=0.0
epochs=40
elayers=12 
batch_size=32 
pretrained_model= 

# decoding parameter
#recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best' 
#recog_model=model.last10.avg.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
#recog_model=model.acc10.avg.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
#decode_dir_affix=acc 

data= 

expdir=exp/track1_accent_classification_${whichformer}_elayers${elayers}

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set_nosp=train
train_set=train_sp
recog_set='dev test'

train_json=$data/$train_set/ar.json
valid_json=$data/dev/ar.json

# extracting filter-bank features and cmvn
if [ $stage -le 0 ] && [ ${stop_stage} -ge 0 ];then
  # speed-perturbed
  utils/perturb_data_dir_speed.sh 0.9 $data/${train_set_nosp} $data/temp1
  utils/perturb_data_dir_speed.sh 1.0 $data/${train_set_nosp} $data/temp2
  utils/perturb_data_dir_speed.sh 1.1 $data/${train_set_nosp} $data/temp3
  utils/combine_data.sh --extra-files utt2uniq $data/${train_set} $data/temp1 $data/temp2 $data/temp3
  rm -r $data/temp1 $data/temp2 $data/temp3

  for i in $train_set $recog_set; do 
    steps/make_fbank_pitch.sh \
      --cmd "$train_cmd" --nj 30 \
      --write_utt2num_frames true \
      --fbank-config conf/fbank_hires.conf \
      $data/$i
    utils/fix_data_dir.sh $data/$i
  done
fi

# generate label file and dump features for olr
if [ $stage -le 1 ] && [ ${stop_stage} -ge 1 ];then
  for data_set in $train_set $recog_set; do
    set_dir=$data/$data_set
    awk '{printf "%s %s\n", $1, $1 }' $set_dir/text > $set_dir/spk2utt.utt
    cp $set_dir/spk2utt.utt $set_dir/utt2spk.utt
    compute-cmvn-stats --spk2utt=ark:$set_dir/spk2utt.utt scp:$set_dir/feats.scp \
      ark,scp:`pwd`/$set_dir/cmvn_utt.ark,$set_dir/cmvn_utt.scp
    local/tools/dump_spk_yzl23.sh --cmd "$train_cmd" --nj 48 \
                                  $set_dir/feats.scp \
                                  $set_dir/cmvn_utt.scp \
                                  exp/dump_feats/$data_set \
                                  $set_dir/dump_utt \
                                  $set_dir/utt2spk.utt
  done
fi

# generate label file for olr
if [ $stage -le 2 ] && [ ${stop_stage} -ge 2 ];then
  for i in $train_set $recog_set;do
    local/tools/data2json.sh \
      --nj 20 --feat $data/$i/dump_utt/feats.scp \
      --text $data/$i/text \
      --oov 3 $data/$i local/files/olr.dict > $data/$i/ar.json
  done
fi

dict=local/files/olr.dict
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "stage 3: Network Training"
  mkdir -p ${expdir}
  run.pl --gpu ${ngpu} ${expdir}/train.log \
    asr_train.py \
    --config ${train_config} \
    --preprocess-conf ${preprocess_config} \
    --ngpu ${ngpu} \
    --backend ${backend} \
    --outdir ${expdir}/results \
    --debugmode ${debugmode} \
    --dict $dict \
    --debugdir ${expdir} \
    --minibatches ${N} \
    --verbose ${verbose} \
    --resume ${resume} \
    --report-interval-iters ${log} \
    --accum-grad ${accum_grad} \
    --n-iter-processes ${n_iter_processes} \
    --elayers ${elayers} \
    --lsm-weight ${lsm_weight} \
    --epochs ${epochs} \
    --batch-size ${batch_size} \
    ${pretrained_model:+--pretrained-model $pretrained_model} \
    --train-json ${train_json} \
    --valid-json ${valid_json}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Averaging"
    average_checkpoints.py --backend pytorch \
                           --snapshots ${expdir}/results/snapshot.ep.* \
                           --out ${expdir}/results/model.acc10.avg.best \
                           --num 10 \
                           --metric acc \
                           --log ${expdir}/results/log
    average_checkpoints.py --backend pytorch \
                           --snapshots ${expdir}/results/snapshot.ep.* \
                           --out ${expdir}/results/model.loss10.avg.best \
                           --num 10 \
                           --metric loss \
                           --log ${expdir}/results/log
fi
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    echo "expdir: $expdir"
    nj=10
    for recog_model in model.acc.best model.acc10.avg.best model.loss10.avg.best; do
      echo "recog_model: $recog_model"
      for test_set in $recog_set; do
        # split data
        decode_dir=decode_${test_set}_${recog_model}
        dev_root=$data/$test_set
        splitjson.py --parts ${nj} ${dev_root}/ar.json
        #### use CPU for decoding
        ngpu=0
        run.pl JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${dev_root}/split${nj}utt/ar.JOB.json \
            --result-label ${expdir}/${decode_dir}/ar.JOB.json \
            --model ${expdir}/results/${recog_model} || exit 1;

        grep YeBobr <(cat $expdir/${decode_dir}/log/decode.[0-9].log $expdir/${decode_dir}/log/decode.10.log) | \
          sed -e 's/YeBobr //g' > $expdir/${decode_dir}/${test_set}_${recog_model}.score

        ./score.sh $expdir/${decode_dir}/${test_set}_${recog_model}.score trails.$test_set;
        echo "Decoding finished: $test_set"
      done
    done
fi
exit 0;

