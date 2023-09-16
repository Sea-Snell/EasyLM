
# 9/11/23

export PYTHONPATH="${PWD}/../:$PYTHONPATH"
conda activate JaxSeq2

export GCP_PATH=young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_data_comparison_1/d496a4bd6cfc4d2692b6aeef34ba731d
export LOCAL_PATH=/shared/csnell/data_study/7B_v2_data
for STEP in 17600
do
    mkdir $LOCAL_PATH/$STEP
    echo "[starting: $LOCAL_PATH/$STEP]"
    gsutil -m cp -r gs://$GCP_PATH/streaming_train_state_$STEP $LOCAL_PATH/$STEP/
    mkdir $LOCAL_PATH/$STEP/pytorch/
    python -m EasyLM.models.llama.convert_easylm_to_hf \
        --load_checkpoint="trainstate_params::$LOCAL_PATH/$STEP/streaming_train_state_$STEP" \
        --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
        --model_size='7b' \
        --output_dir="$LOCAL_PATH/$STEP/pytorch"
    echo "[finished: $LOCAL_PATH/$STEP]"
done

export GCP_PATH=young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_data_comparison_1/39e8cadf271246b5bad93b0da3bd96c3
export LOCAL_PATH=/shared/csnell/data_study/7B_v1_data
for STEP in 22000
do
    mkdir $LOCAL_PATH/$STEP
    echo "[starting: $LOCAL_PATH/$STEP]"
    gsutil -m cp -r gs://$GCP_PATH/streaming_train_state_$STEP $LOCAL_PATH/$STEP/
    mkdir $LOCAL_PATH/$STEP/pytorch/
    python -m EasyLM.models.llama.convert_easylm_to_hf \
        --load_checkpoint="trainstate_params::$LOCAL_PATH/$STEP/streaming_train_state_$STEP" \
        --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
        --model_size='7b' \
        --output_dir="$LOCAL_PATH/$STEP/pytorch"
    echo "[finished: $LOCAL_PATH/$STEP]"
done

# 9/8/23

# export PYTHONPATH="${PWD}/../:$PYTHONPATH"
# conda activate JaxSeq2

# export GCP_PATH=young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_data_comparison_1/d496a4bd6cfc4d2692b6aeef34ba731d
# export LOCAL_PATH=/shared/csnell/data_study/7B_v2_data
# for STEP in 22000 17600 13200 8800 4400
# do
#     mkdir $LOCAL_PATH/$STEP
#     echo "[starting: $LOCAL_PATH/$STEP]"
#     gsutil -m cp -r gs://$GCP_PATH/streaming_train_state_$STEP $LOCAL_PATH/$STEP/
#     mkdir $LOCAL_PATH/$STEP/pytorch/
#     python -m EasyLM.models.llama.convert_easylm_to_hf \
#         --load_checkpoint="trainstate_params::$LOCAL_PATH/$STEP/streaming_train_state_$STEP" \
#         --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#         --model_size='7b' \
#         --output_dir="$LOCAL_PATH/$STEP/pytorch"
#     echo "[finished: $LOCAL_PATH/$STEP]"
# done

# export GCP_PATH=young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_data_comparison_1/39e8cadf271246b5bad93b0da3bd96c3
# export LOCAL_PATH=/shared/csnell/data_study/7B_v1_data
# for STEP in 22000 17600 13200 8800 4400
# do
#     mkdir $LOCAL_PATH/$STEP
#     echo "[starting: $LOCAL_PATH/$STEP]"
#     gsutil -m cp -r gs://$GCP_PATH/streaming_train_state_$STEP $LOCAL_PATH/$STEP/
#     mkdir $LOCAL_PATH/$STEP/pytorch/
#     python -m EasyLM.models.llama.convert_easylm_to_hf \
#         --load_checkpoint="trainstate_params::$LOCAL_PATH/$STEP/streaming_train_state_$STEP" \
#         --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#         --model_size='7b' \
#         --output_dir="$LOCAL_PATH/$STEP/pytorch"
#     echo "[finished: $LOCAL_PATH/$STEP]"
# done


# 8/25/23

# export PYTHONPATH="${PWD}/../:$PYTHONPATH"
# conda activate JaxSeq2

# # 3B_v1

# export GCP_PATH=young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_2/97bbde1931b94ba1a4ec66fa9b80b37a
# export LOCAL_PATH=/shared/csnell/openllama/3B_v1
# export STEP=10000
# mkdir $LOCAL_PATH/$STEP
# echo "[starting: $LOCAL_PATH/$STEP]"
# gsutil -m cp -r gs://$GCP_PATH/streaming_train_state_$STEP $LOCAL_PATH/$STEP/
# mkdir $LOCAL_PATH/$STEP/pytorch/
# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint="trainstate_params::$LOCAL_PATH/$STEP/streaming_train_state_$STEP" \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='3b' \
#     --output_dir="$LOCAL_PATH/$STEP/pytorch"
# echo "[finished: $LOCAL_PATH/$STEP]"

# export GCP_PATH=young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_2/3ac40d6a66c742bcbe281491b9c404be
# export LOCAL_PATH=/shared/csnell/openllama/3B_v1
# for STEP in 50000 100000 150000 200000 250000
# do
#     mkdir $LOCAL_PATH/$STEP
#     echo "[starting: $LOCAL_PATH/$STEP]"
#     gsutil -m cp -r gs://$GCP_PATH/streaming_train_state_$STEP $LOCAL_PATH/$STEP/
#     mkdir $LOCAL_PATH/$STEP/pytorch/
#     python -m EasyLM.models.llama.convert_easylm_to_hf \
#         --load_checkpoint="trainstate_params::$LOCAL_PATH/$STEP/streaming_train_state_$STEP" \
#         --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#         --model_size='3b' \
#         --output_dir="$LOCAL_PATH/$STEP/pytorch"
#     echo "[finished: $LOCAL_PATH/$STEP]"
# done

# # 7B_v1

# export GCP_PATH=young-nlp-us-c2/experiment_output/young/easy_lm/open_llama_2/77aec5c3e8774c26822503302cd51f1a
# export LOCAL_PATH=/shared/csnell/openllama/7B_v1
# export STEP=10000
# mkdir $LOCAL_PATH/$STEP
# echo "[starting: $LOCAL_PATH/$STEP]"
# gsutil -m cp -r gs://$GCP_PATH/streaming_train_state_$STEP $LOCAL_PATH/$STEP/
# mkdir $LOCAL_PATH/$STEP/pytorch/
# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint="trainstate_params::$LOCAL_PATH/$STEP/streaming_train_state_$STEP" \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='7b' \
#     --output_dir="$LOCAL_PATH/$STEP/pytorch"
# echo "[finished: $LOCAL_PATH/$STEP]"

# export GCP_PATH=young-nlp-us-c2/experiment_output/young/easy_lm/open_llama_2/47c4ea0167d744f4b2cacddff0e5e750
# export LOCAL_PATH=/shared/csnell/openllama/7B_v1
# export STEP=50000
# mkdir $LOCAL_PATH/$STEP
# echo "[starting: $LOCAL_PATH/$STEP]"
# gsutil -m cp -r gs://$GCP_PATH/streaming_train_state_$STEP $LOCAL_PATH/$STEP/
# mkdir $LOCAL_PATH/$STEP/pytorch/
# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint="trainstate_params::$LOCAL_PATH/$STEP/streaming_train_state_$STEP" \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='7b' \
#     --output_dir="$LOCAL_PATH/$STEP/pytorch"
# echo "[finished: $LOCAL_PATH/$STEP]"

# export GCP_PATH=young-nlp-us-c2/experiment_output/young/easy_lm/open_llama_2/d8fa876cf63f4aac864e801476962da9
# export LOCAL_PATH=/shared/csnell/openllama/7B_v1
# export STEP=100000
# mkdir $LOCAL_PATH/$STEP
# echo "[starting: $LOCAL_PATH/$STEP]"
# gsutil -m cp -r gs://$GCP_PATH/streaming_train_state_$STEP $LOCAL_PATH/$STEP/
# mkdir $LOCAL_PATH/$STEP/pytorch/
# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint="trainstate_params::$LOCAL_PATH/$STEP/streaming_train_state_$STEP" \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='7b' \
#     --output_dir="$LOCAL_PATH/$STEP/pytorch"
# echo "[finished: $LOCAL_PATH/$STEP]"

# export GCP_PATH=young-nlp-us-c2/experiment_output/young/easy_lm/open_llama_2/5d1723463be04d0b9688fdd434adb620
# export LOCAL_PATH=/shared/csnell/openllama/7B_v1
# for STEP in 150000 200000
# do
#     mkdir $LOCAL_PATH/$STEP
#     echo "[starting: $LOCAL_PATH/$STEP]"
#     gsutil -m cp -r gs://$GCP_PATH/streaming_train_state_$STEP $LOCAL_PATH/$STEP/
#     mkdir $LOCAL_PATH/$STEP/pytorch/
#     python -m EasyLM.models.llama.convert_easylm_to_hf \
#         --load_checkpoint="trainstate_params::$LOCAL_PATH/$STEP/streaming_train_state_$STEP" \
#         --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#         --model_size='7b' \
#         --output_dir="$LOCAL_PATH/$STEP/pytorch"
#     echo "[finished: $LOCAL_PATH/$STEP]"
# done

# export GCP_PATH=young-nlp-us-c2/experiment_output/young/easy_lm/open_llama_2/edca7640bb4345f780eb8ad3c2ceec7f
# export LOCAL_PATH=/shared/csnell/openllama/7B_v1
# export STEP=250000
# mkdir $LOCAL_PATH/$STEP
# echo "[starting: $LOCAL_PATH/$STEP]"
# gsutil -m cp -r gs://$GCP_PATH/streaming_train_state_$STEP $LOCAL_PATH/$STEP/
# mkdir $LOCAL_PATH/$STEP/pytorch/
# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint="trainstate_params::$LOCAL_PATH/$STEP/streaming_train_state_$STEP" \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='7b' \
#     --output_dir="$LOCAL_PATH/$STEP/pytorch"
# echo "[finished: $LOCAL_PATH/$STEP]"

# # 13B_v1

# export GCP_PATH=young-nlp-us-c2/experiment_output/young/easy_lm/open_llama_13b_1/0565896c9d674ac190db05d5ea452bdd
# export LOCAL_PATH=/shared/csnell/openllama/13B_v1
# for STEP in 20000 100000
# do
#     mkdir $LOCAL_PATH/$STEP
#     echo "[starting: $LOCAL_PATH/$STEP]"
#     gsutil -m cp -r gs://$GCP_PATH/streaming_train_state_$STEP $LOCAL_PATH/$STEP/
#     mkdir $LOCAL_PATH/$STEP/pytorch/
#     python -m EasyLM.models.llama.convert_easylm_to_hf \
#         --load_checkpoint="trainstate_params::$LOCAL_PATH/$STEP/streaming_train_state_$STEP" \
#         --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#         --model_size='13b' \
#         --output_dir="$LOCAL_PATH/$STEP/pytorch"
#     echo "[finished: $LOCAL_PATH/$STEP]"
# done

# export GCP_PATH=young-nlp-us-c2/experiment_output/young/easy_lm/open_llama_13b_1/e2347d1ef9d641fd98b09ec80719dad7
# export LOCAL_PATH=/shared/csnell/openllama/13B_v1
# for STEP in 200000 300000
# do
#     mkdir $LOCAL_PATH/$STEP
#     echo "[starting: $LOCAL_PATH/$STEP]"
#     gsutil -m cp -r gs://$GCP_PATH/streaming_train_state_$STEP $LOCAL_PATH/$STEP/
#     mkdir $LOCAL_PATH/$STEP/pytorch/
#     python -m EasyLM.models.llama.convert_easylm_to_hf \
#         --load_checkpoint="trainstate_params::$LOCAL_PATH/$STEP/streaming_train_state_$STEP" \
#         --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#         --model_size='13b' \
#         --output_dir="$LOCAL_PATH/$STEP/pytorch"
#     echo "[finished: $LOCAL_PATH/$STEP]"
# done

# export GCP_PATH=young-nlp-us-c2/experiment_output/young/easy_lm/open_llama_13b_1/4ba193e6f84340b4b56ded1fd046789d
# export LOCAL_PATH=/shared/csnell/openllama/13B_v1
# for STEP in 400000 500000
# do
#     mkdir $LOCAL_PATH/$STEP
#     echo "[starting: $LOCAL_PATH/$STEP]"
#     gsutil -m cp -r gs://$GCP_PATH/streaming_train_state_$STEP $LOCAL_PATH/$STEP/
#     mkdir $LOCAL_PATH/$STEP/pytorch/
#     python -m EasyLM.models.llama.convert_easylm_to_hf \
#         --load_checkpoint="trainstate_params::$LOCAL_PATH/$STEP/streaming_train_state_$STEP" \
#         --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#         --model_size='13b' \
#         --output_dir="$LOCAL_PATH/$STEP/pytorch"
#     echo "[finished: $LOCAL_PATH/$STEP]"
# done


# 8/22/23

# export PYTHONPATH="${PWD}/../:$PYTHONPATH"
# conda activate JaxSeq2

# mkdir /shared/csnell/data_study/1B_v1_data/9800/pytorch/
# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint='trainstate_params::/shared/csnell/data_study/1B_v1_data/9800/streaming_train_state_9800' \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='1b' \
#     --output_dir='/shared/csnell/data_study/1B_v1_data/9800/pytorch'

# mkdir /shared/csnell/data_study/1B_v1_data/5600/pytorch/
# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint='trainstate_params::/shared/csnell/data_study/1B_v1_data/5600/streaming_train_state_5600' \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='1b' \
#     --output_dir='/shared/csnell/data_study/1B_v1_data/5600/pytorch'

# mkdir /shared/csnell/data_study/1B_v1_data/1400/pytorch/
# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint='trainstate_params::/shared/csnell/data_study/1B_v1_data/1400/streaming_train_state_1400' \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='1b' \
#     --output_dir='/shared/csnell/data_study/1B_v1_data/1400/pytorch'


# mkdir /shared/csnell/data_study/1B_v2_data/9800/pytorch/
# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint='trainstate_params::/shared/csnell/data_study/1B_v2_data/9800/streaming_train_state_9800' \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='1b' \
#     --output_dir='/shared/csnell/data_study/1B_v2_data/9800/pytorch'

# mkdir /shared/csnell/data_study/1B_v2_data/5600/pytorch/
# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint='trainstate_params::/shared/csnell/data_study/1B_v2_data/5600/streaming_train_state_5600' \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='1b' \
#     --output_dir='/shared/csnell/data_study/1B_v2_data/5600/pytorch'

# mkdir /shared/csnell/data_study/1B_v2_data/1400/pytorch/
# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint='trainstate_params::/shared/csnell/data_study/1B_v2_data/1400/streaming_train_state_1400' \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='1b' \
#     --output_dir='/shared/csnell/data_study/1B_v2_data/1400/pytorch'



# mkdir /shared/csnell/data_study/3B_v1_data/35200/pytorch/
# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint='trainstate_params::/shared/csnell/data_study/3B_v1_data/35200/streaming_train_state_35200' \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='3b' \
#     --output_dir='/shared/csnell/data_study/3B_v1_data/35200/pytorch'

# mkdir /shared/csnell/data_study/3B_v1_data/28800/pytorch/
# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint='trainstate_params::/shared/csnell/data_study/3B_v1_data/28800/streaming_train_state_28800' \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='3b' \
#     --output_dir='/shared/csnell/data_study/3B_v1_data/28800/pytorch'

# mkdir /shared/csnell/data_study/3B_v1_data/22400/pytorch/
# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint='trainstate_params::/shared/csnell/data_study/3B_v1_data/22400/streaming_train_state_22400' \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='3b' \
#     --output_dir='/shared/csnell/data_study/3B_v1_data/22400/pytorch'


# mkdir /shared/csnell/data_study/3B_v2_data/30800/pytorch/
# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint='trainstate_params::/shared/csnell/data_study/3B_v2_data/30800/streaming_train_state_30800' \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='3b' \
#     --output_dir='/shared/csnell/data_study/3B_v2_data/30800/pytorch'

# mkdir /shared/csnell/data_study/3B_v2_data/17600/pytorch/
# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint='trainstate_params::/shared/csnell/data_study/3B_v2_data/17600/streaming_train_state_17600' \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='3b' \
#     --output_dir='/shared/csnell/data_study/3B_v2_data/17600/pytorch'

# mkdir /shared/csnell/data_study/3B_v2_data/4400/pytorch/
# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint='trainstate_params::/shared/csnell/data_study/3B_v2_data/4400/streaming_train_state_4400' \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='3b' \
#     --output_dir='/shared/csnell/data_study/3B_v2_data/4400/pytorch'


# mkdir /shared/csnell/data_study/1B_v1_data/9800/
# gsutil -m cp -r gs://young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_data_comparison_1/60163791038f4d63b28e1c08381ac57e/streaming_train_state_9800 /shared/csnell/data_study/1B_v1_data/9800/

# mkdir /shared/csnell/data_study/1B_v1_data/5600/
# gsutil -m cp -r gs://young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_data_comparison_1/60163791038f4d63b28e1c08381ac57e/streaming_train_state_5600 /shared/csnell/data_study/1B_v1_data/5600/

# mkdir /shared/csnell/data_study/1B_v1_data/1400/
# gsutil -m cp -r gs://young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_data_comparison_1/60163791038f4d63b28e1c08381ac57e/streaming_train_state_1400 /shared/csnell/data_study/1B_v1_data/1400/



# mkdir /shared/csnell/data_study/1B_v2_data/9800/
# gsutil -m cp -r gs://young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_data_comparison_1/c0439e5398724262a657ce378800dae8/streaming_train_state_9800 /shared/csnell/data_study/1B_v2_data/9800/

# mkdir /shared/csnell/data_study/1B_v2_data/5600/
# gsutil -m cp -r gs://young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_data_comparison_1/c0439e5398724262a657ce378800dae8/streaming_train_state_5600 /shared/csnell/data_study/1B_v2_data/5600/

# mkdir /shared/csnell/data_study/1B_v2_data/1400/
# gsutil -m cp -r gs://young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_data_comparison_1/c0439e5398724262a657ce378800dae8/streaming_train_state_1400 /shared/csnell/data_study/1B_v2_data/1400/



# mkdir /shared/csnell/data_study/3B_v1_data/35200/
# gsutil -m cp -r gs://young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_data_comparison_1/00fc1abd378143e1b5ebe38e1d3d4e61/streaming_train_state_35200 /shared/csnell/data_study/3B_v1_data/35200/

# mkdir /shared/csnell/data_study/3B_v1_data/28800/
# gsutil -m cp -r gs://young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_data_comparison_1/00fc1abd378143e1b5ebe38e1d3d4e61/streaming_train_state_28800 /shared/csnell/data_study/3B_v1_data/28800/

# mkdir /shared/csnell/data_study/3B_v1_data/22400/
# gsutil -m cp -r gs://young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_data_comparison_1/00fc1abd378143e1b5ebe38e1d3d4e61/streaming_train_state_22400 /shared/csnell/data_study/3B_v1_data/22400/



# mkdir /shared/csnell/data_study/3B_v2_data/30800/
# gsutil -m cp -r gs://young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_data_comparison_1/31359782f4de4027875bc266b0071d13/streaming_train_state_30800 /shared/csnell/data_study/3B_v2_data/30800/

# mkdir /shared/csnell/data_study/3B_v2_data/17600/
# gsutil -m cp -r gs://young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_data_comparison_1/31359782f4de4027875bc266b0071d13/streaming_train_state_17600 /shared/csnell/data_study/3B_v2_data/17600/

# mkdir /shared/csnell/data_study/3B_v2_data/4400/
# gsutil -m cp -r gs://young-nlp-us-e1/experiment_output/young/easy_lm/open_llama_data_comparison_1/31359782f4de4027875bc266b0071d13/streaming_train_state_4400 /shared/csnell/data_study/3B_v2_data/4400/



# old

# export PYTHONPATH="${PWD}/../:$PYTHONPATH"
# conda activate JaxSeq2

# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint='trainstate_params::/shared/csnell/data_study/1B_v2_data/streaming_train_state_14000' \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='1b' \
#     --output_dir='/shared/csnell/data_study/1B_v2_data/pytorch'

# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint='trainstate_params::/shared/csnell/data_study/3B_v1_data/streaming_train_state_44000' \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='3b' \
#     --output_dir='/shared/csnell/data_study/3B_v1_data/pytorch'

# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint='trainstate_params::/shared/csnell/data_study/3B_v2_data/streaming_train_state_44000' \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='3b' \
#     --output_dir='/shared/csnell/data_study/3B_v2_data/pytorch'

# python -m EasyLM.models.llama.convert_easylm_to_hf \
#     --load_checkpoint='trainstate_params::/shared/csnell/data_study/1B_v1_data/streaming_train_state_14000' \
#     --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
#     --model_size='1b' \
#     --output_dir='/shared/csnell/data_study/1B_v1_data/pytorch'


