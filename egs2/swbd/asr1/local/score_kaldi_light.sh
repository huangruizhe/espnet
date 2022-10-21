#!/usr/bin/env bash

# https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/steps/scoring/score_kaldi_wer.sh

[ -f ./path.sh ] && . ./path.sh

. parse_options.sh || exit 1;

ref_text=$1
hyp_text=$2
data_dir=$3
decode_dir=$4

ref_filtering_cmd="cat"
[ -x local/wer_output_filter ] && ref_filtering_cmd="local/wer_output_filter"
[ -x local/wer_ref_filter ] && ref_filtering_cmd="local/wer_ref_filter"
hyp_filtering_cmd="cat"
[ -x local/wer_output_filter ] && hyp_filtering_cmd="local/wer_output_filter"
[ -x local/wer_hyp_filter ] && hyp_filtering_cmd="local/wer_hyp_filter"

echo "ref_filtering_cmd=${ref_filtering_cmd}"
echo "hyp_filtering_cmd=${hyp_filtering_cmd}"

mkdir -p $decode_dir/scoring_kaldi/
cat $ref_text | $ref_filtering_cmd > $decode_dir/scoring_kaldi/ref.text
cat $hyp_text | $hyp_filtering_cmd > $decode_dir/scoring_kaldi/hyp.text

# cat $hyp_text | $hyp_filtering_cmd | \
#     compute-wer --text --mode=present \
#     ark:"cat $ref_text | $ref_filtering_cmd |"  ark,p:- \
#     > $decode_dir/scoring_kaldi/wer

cat $decode_dir/scoring_kaldi/hyp.text | \
    compute-wer --text --mode=present \
    ark:$decode_dir/scoring_kaldi/ref.text ark,p:- \
    > $decode_dir/scoring_kaldi/wer || exit 1;


mkdir -p $decode_dir/scoring_kaldi/wer_details

cat $decode_dir/scoring_kaldi/hyp.text | \
    align-text --special-symbol="'***'" ark:$decode_dir/scoring_kaldi/ref.text ark:- ark,t:- |  \
    utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" | tee $decode_dir/scoring_kaldi/wer_details/per_utt | \
    utils/scoring/wer_per_spk_details.pl $data_dir/utt2spk > $decode_dir/scoring_kaldi/wer_details/per_spk || exit 1;

cat $decode_dir/scoring_kaldi/wer_details/per_utt | \
    utils/scoring/wer_ops_details.pl --special-symbol "'***'" | \
    sort -b -i -k 1,1 -k 4,4rn -k 2,2 -k 3,3 > $decode_dir/scoring_kaldi/wer_details/ops || exit 1;

compute-wer-bootci --mode=present \
    ark:$decode_dir/scoring_kaldi/ref.text ark:$decode_dir/scoring_kaldi/hyp.text \
    > $decode_dir/scoring_kaldi/wer_details/wer_bootci || exit 1;

cat $decode_dir/scoring_kaldi/wer
echo "Done:" $decode_dir/scoring_kaldi/

exit 0;

# Usages:

# std2006_dev
ref=/export/fs04/a12/rhuang/kws/kws_exp/shay/s5c/exp/chain/tdnn7r_sp/decode_std2006_dev_sw1_fsh_fg_rnnlm_1e_0.45/scoring_kaldi/test_filt.txt
hyp=exp/Yuekai_Zhang/swbd_asr_train_asr_cformer5_raw_bpe2000_sp_valid.acc.ave/decode_asr_lm_lm_train_lm_bpe2000_valid.loss.best_asr_model_valid.acc.ave/std2006_dev/text
data=data/std2006_dev/
decode=exp/Yuekai_Zhang/swbd_asr_train_asr_cformer5_raw_bpe2000_sp_valid.acc.ave/decode_asr_lm_lm_train_lm_bpe2000_valid.loss.best_asr_model_valid.acc.ave/std2006_dev/

# std2006_eval
ref=/export/fs04/a12/rhuang/kws/kws_exp/shay/s5c/exp/chain/tdnn7r_sp/decode_std2006_eval_sw1_fsh_fg_rnnlm_1e_0.45/scoring_kaldi/test_filt.txt
hyp=exp/Yuekai_Zhang/swbd_asr_train_asr_cformer5_raw_bpe2000_sp_valid.acc.ave/decode_asr_lm_lm_train_lm_bpe2000_valid.loss.best_asr_model_valid.acc.ave/std2006_eval/text
data=data/std2006_eval/
decode=exp/Yuekai_Zhang/swbd_asr_train_asr_cformer5_raw_bpe2000_sp_valid.acc.ave/decode_asr_lm_lm_train_lm_bpe2000_valid.loss.best_asr_model_valid.acc.ave/std2006_eval/

# std2006_eval (kaldi)
ref=/export/fs04/a12/rhuang/kws/kws_exp/shay/s5c/exp/chain/tdnn7r_sp/decode_std2006_eval_sw1_fsh_fg_rnnlm_1e_0.45/scoring_kaldi/test_filt.txt
hyp=/export/fs04/a12/rhuang/kws/kws_exp/shay/s5c/exp/chain/tdnn7r_sp/decode_std2006_eval_sw1_fsh_fg_rnnlm_1e_0.45/scoring_kaldi/penalty_0.0/12.txt
data=data/std2006_dev/
decode=tmp

# callhome train
ref=/export/fs04/a12/rhuang/kws/kws_exp/shay/s5c/exp/chain/tdnn7r_sp/decode_callhome_train_sw1_fsh_fg_rnnlm_1e_0.45/scoring_kaldi/test_filt.txt
hyp=exp/Yuekai_Zhang/swbd_asr_train_asr_cformer5_raw_bpe2000_sp_valid.acc.ave/decode_asr_lm_lm_train_lm_bpe2000_valid.loss.best_asr_model_valid.acc.ave/callhome_train/text
data=data/callhome_train/
decode=exp/Yuekai_Zhang/swbd_asr_train_asr_cformer5_raw_bpe2000_sp_valid.acc.ave/decode_asr_lm_lm_train_lm_bpe2000_valid.loss.best_asr_model_valid.acc.ave/callhome_train/

# callhome dev
ref=/export/fs04/a12/rhuang/kws/kws_exp/shay/s5c/exp/chain/tdnn7r_sp/decode_callhome_dev_sw1_fsh_fg_rnnlm_1e_0.45/scoring_kaldi/test_filt.txt
hyp=exp/Yuekai_Zhang/swbd_asr_train_asr_cformer5_raw_bpe2000_sp_valid.acc.ave/decode_asr_lm_lm_train_lm_bpe2000_valid.loss.best_asr_model_valid.acc.ave/callhome_dev/text
data=data/callhome_dev/
decode=exp/Yuekai_Zhang/swbd_asr_train_asr_cformer5_raw_bpe2000_sp_valid.acc.ave/decode_asr_lm_lm_train_lm_bpe2000_valid.loss.best_asr_model_valid.acc.ave/callhome_dev/

# callhome eval
ref=/export/fs04/a12/rhuang/kws/kws_exp/shay/s5c/exp/chain/tdnn7r_sp/decode_callhome_eval_sw1_fsh_fg_rnnlm_1e_0.45/scoring_kaldi/test_filt.txt
hyp=exp/Yuekai_Zhang/swbd_asr_train_asr_cformer5_raw_bpe2000_sp_valid.acc.ave/decode_asr_lm_lm_train_lm_bpe2000_valid.loss.best_asr_model_valid.acc.ave/callhome_eval/text
data=data/callhome_eval/
decode=exp/Yuekai_Zhang/swbd_asr_train_asr_cformer5_raw_bpe2000_sp_valid.acc.ave/decode_asr_lm_lm_train_lm_bpe2000_valid.loss.best_asr_model_valid.acc.ave/callhome_eval/

# eval2000
ref=/export/fs04/a12/rhuang/kws/kws_exp/shay/s5c/exp/chain/tdnn7r_sp/decode_eval2000_sw1_fsh_fg_rnnlm_1e_0.45/score_by_kaldi_with_wer_filter/scoring_kaldi/test_filt.txt
hyp=exp/Yuekai_Zhang/swbd_asr_train_asr_cformer5_raw_bpe2000_sp_valid.acc.ave/decode_asr_lm_lm_train_lm_bpe2000_valid.loss.best_asr_model_valid.acc.ave/eval2000/text
data=data/eval2000/
decode=exp/Yuekai_Zhang/swbd_asr_train_asr_cformer5_raw_bpe2000_sp_valid.acc.ave/decode_asr_lm_lm_train_lm_bpe2000_valid.loss.best_asr_model_valid.acc.ave/eval2000/

# en_ or sw_
grep -E "en_.*raw" $decode/scoring_kaldi/wer_details/per_spk | \
    tr -s ' ' | cut -d' ' -f4-9 | \
    awk -F " " '{s1=s1+$1;s2=s2+$2;s3=s3+$3;s4=s4+$4;s5=s5+$5;s6=s6+$6;} END{print "     #WORD   Corr    Sub    Ins    Del    Err"; print "Total "s1     " "s2" "s4" "s5" "s3" "s6; print "Total "s1/s1" "s2/s1"cor "s4/s1"ins "s5/s1"del "s3/s1"sub "s6/s1}'

bash local/score_kaldi_light.sh $ref $hyp $data $decode
