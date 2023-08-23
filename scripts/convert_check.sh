
# 8/22/23

export PYTHONPATH="${PWD}/../:$PYTHONPATH"
conda activate JaxSeq2

mkdir /shared/csnell/data_study/1B_v1_data/9800/pytorch/
python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='trainstate_params::/shared/csnell/data_study/1B_v1_data/9800/streaming_train_state_9800' \
    --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
    --model_size='1b' \
    --output_dir='/shared/csnell/data_study/1B_v1_data/9800/pytorch'

mkdir /shared/csnell/data_study/1B_v1_data/5600/pytorch/
python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='trainstate_params::/shared/csnell/data_study/1B_v1_data/5600/streaming_train_state_5600' \
    --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
    --model_size='1b' \
    --output_dir='/shared/csnell/data_study/1B_v1_data/5600/pytorch'

mkdir /shared/csnell/data_study/1B_v1_data/1400/pytorch/
python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='trainstate_params::/shared/csnell/data_study/1B_v1_data/1400/streaming_train_state_1400' \
    --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
    --model_size='1b' \
    --output_dir='/shared/csnell/data_study/1B_v1_data/1400/pytorch'


mkdir /shared/csnell/data_study/1B_v2_data/9800/pytorch/
python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='trainstate_params::/shared/csnell/data_study/1B_v2_data/9800/streaming_train_state_9800' \
    --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
    --model_size='1b' \
    --output_dir='/shared/csnell/data_study/1B_v2_data/9800/pytorch'

mkdir /shared/csnell/data_study/1B_v2_data/5600/pytorch/
python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='trainstate_params::/shared/csnell/data_study/1B_v2_data/5600/streaming_train_state_5600' \
    --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
    --model_size='1b' \
    --output_dir='/shared/csnell/data_study/1B_v2_data/5600/pytorch'

mkdir /shared/csnell/data_study/1B_v2_data/1400/pytorch/
python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='trainstate_params::/shared/csnell/data_study/1B_v2_data/1400/streaming_train_state_1400' \
    --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
    --model_size='1b' \
    --output_dir='/shared/csnell/data_study/1B_v2_data/1400/pytorch'



mkdir /shared/csnell/data_study/3B_v1_data/35200/pytorch/
python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='trainstate_params::/shared/csnell/data_study/3B_v1_data/35200/streaming_train_state_35200' \
    --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
    --model_size='3b' \
    --output_dir='/shared/csnell/data_study/3B_v1_data/35200/pytorch'

mkdir /shared/csnell/data_study/3B_v1_data/28800/pytorch/
python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='trainstate_params::/shared/csnell/data_study/3B_v1_data/28800/streaming_train_state_28800' \
    --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
    --model_size='3b' \
    --output_dir='/shared/csnell/data_study/3B_v1_data/28800/pytorch'

mkdir /shared/csnell/data_study/3B_v1_data/22400/pytorch/
python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='trainstate_params::/shared/csnell/data_study/3B_v1_data/22400/streaming_train_state_22400' \
    --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
    --model_size='3b' \
    --output_dir='/shared/csnell/data_study/3B_v1_data/22400/pytorch'


mkdir /shared/csnell/data_study/3B_v2_data/30800/pytorch/
python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='trainstate_params::/shared/csnell/data_study/3B_v2_data/30800/streaming_train_state_30800' \
    --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
    --model_size='3b' \
    --output_dir='/shared/csnell/data_study/3B_v2_data/30800/pytorch'

mkdir /shared/csnell/data_study/3B_v2_data/17600/pytorch/
python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='trainstate_params::/shared/csnell/data_study/3B_v2_data/17600/streaming_train_state_17600' \
    --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
    --model_size='3b' \
    --output_dir='/shared/csnell/data_study/3B_v2_data/17600/pytorch'

mkdir /shared/csnell/data_study/3B_v2_data/4400/pytorch/
python -m EasyLM.models.llama.convert_easylm_to_hf \
    --load_checkpoint='trainstate_params::/shared/csnell/data_study/3B_v2_data/4400/streaming_train_state_4400' \
    --tokenizer_path='/shared/csnell/openllama_tokenizer/open_llama_3.model' \
    --model_size='3b' \
    --output_dir='/shared/csnell/data_study/3B_v2_data/4400/pytorch'


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


