import dataclasses
import pprint
from functools import partial
import re

from tqdm import tqdm, trange
import numpy as np
import mlxu

import jax
import jax.numpy as jnp
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
import flax
from flax import linen as nn
from flax.jax_utils import prefetch_to_device
from flax.training.train_state import TrainState
import optax

from EasyLM.data import DatasetFactory
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, named_tree_map, global_norm,
    set_random_seed, average_metrics, get_weight_decay_mask,
    make_shard_and_gather_fns, with_sharding_constraint
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfig, FlaxLLaMAForCausalLM, FlaxLLaMAForCausalLMModule
)


from JaxSeq.utils import convert_path, MapIterable, FileOpenIterable
from JaxSeq.utils import jsonl_stream
from JaxSeq.models.llama.load import load_tokenizer
from JaxSeq.data import MaskIterableDataset

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    initialize_jax_distributed=False,
    mesh_dim='1,-1,1',
    total_steps=10000,
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    load_dataset_state='',
    log_freq=50,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_steps=0,
    tokenizer=LLaMAConfig.get_tokenizer_config(),
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfig.get_default_config(),
    logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
)


def main(argv):
    elm_tokenizer = LLaMAConfig.get_tokenizer(FLAGS.tokenizer)
    elm_dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, elm_tokenizer)

    tokenizer_path = '/shared/csnell/llama_weights/tokenizer.model'
    train_data_path = '/shared/csnell/datasets/JaxSeq_chat_koala/chat_data_v3_train.jsonl'
    max_length = 2048

    tokenizer = load_tokenizer(
        tokenizer_path, 
        bos_token="<s>", 
        eos_token="</s>", 
        add_bos_token=False, 
        add_eos_token=False, 
    )
    tokenizer.pad_token_id = None # set pad token to None during training
    tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>'})

    train_data = MaskIterableDataset.packed_from_str_segments_iterable(
        MapIterable(lambda x: [(tokenizer.bos_token, 0.0)]+x+[(tokenizer.eos_token, 1.0)], FileOpenIterable(convert_path(train_data_path), 'r', pipe=jsonl_stream)), 
        tokenizer, 
        max_length=max_length, 
        truncate=True, 
        buffer_start_str=None, 
    )

    for (a, _), b in zip(elm_dataset, train_data):
        import IPython; IPython.embed()

if __name__ == "__main__":
    mlxu.run(main)
