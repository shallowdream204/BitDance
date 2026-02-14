import functools
import os
import gc
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
    ShardedStateDictConfig,
)
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, load, save
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import load_file, save_file

from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer  # NEW
from modeling.vision_head.flow_head_parallel_x import TransEncoder
from modeling.utils import MLPconnector


import shutil
import glob
from utils.fs import copy as fs_copy, mkdir as fs_mkdir, move as fs_move


def get_shard_info(fsdp_config):
    """Return (shard_index, total_shards) for the given sharding strategy."""
    strategy = fsdp_config.sharding_strategy.upper()
    rank, world_size = dist.get_rank(), dist.get_world_size()

    if strategy in ("FULL_SHARD", "SHARD_GRAD_OP"):
        return rank, world_size
    if strategy == "HYBRID_SHARD":
        return rank % fsdp_config.num_shard, fsdp_config.num_shard
    if strategy == "NO_SHARD":
        return 0, 1
    raise NotImplementedError(f"Unknown sharding strategy: {strategy}")


def _should_save_optimizer_for_rank(fsdp_config, rank):
    """Return True if this rank should write the optimizer shard file."""
    strategy = fsdp_config.sharding_strategy.upper()
    if strategy in ("FULL_SHARD", "SHARD_GRAD_OP"):
        return True 
    if strategy == "NO_SHARD":
        return rank == 0
    if strategy == "HYBRID_SHARD":
        return rank < fsdp_config.num_shard
    raise NotImplementedError(f"Unknown sharding strategy: {fsdp_config.sharding_strategy}")


def _optimizer_path(save_dir, shard_index, total_shards):
    return os.path.join(save_dir, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt")


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy,
        backward_prefetch,
        cpu_offload,
        num_replicate,
        num_shard=8,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard


def fsdp_wrapper(original_model, fsdp_config, ignored_modules=None):
    ignored_modules = ignored_modules or []   # [model.model.embed_tokens]  # optional
    device_mesh = None
    if fsdp_config.sharding_strategy.upper() == "HYBRID_SHARD":
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard"),
        )

    return FSDP(
        original_model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen3DecoderLayer, TransEncoder, MLPconnector},
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
    )


class FSDPCheckpoint:
    @staticmethod
    def _save_optimizer_if_needed(fsdp_config, optimizer, save_path, shard_index, total_shards):
        rank = dist.get_rank()
        if _should_save_optimizer_for_rank(fsdp_config, rank):
            opt_path = _optimizer_path(save_path, shard_index, total_shards)
            torch.save(optimizer.state_dict(), opt_path)

    @staticmethod
    def fsdp_save_ckpt(
        ckpt_dir,
        train_steps,
        model,
        optimizer,
        scheduler,
        data_status,
        logger,
        fsdp_config,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        # Save full model on rank 0 (offload to cpu)
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            model_state_dict = model.state_dict()
            if dist.get_rank() == 0:
                save_file(model_state_dict, os.path.join(save_path, "model.safetensors"))
                del model_state_dict
                gc.collect()
                torch.cuda.empty_cache()

        # Save local optimizer shards according to strategy
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            shard_index, total_shards = get_shard_info(fsdp_config)
            FSDPCheckpoint._save_optimizer_if_needed(
                fsdp_config, optimizer, save_path, shard_index, total_shards
            )

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if data_status is not None:
            ds_dir = os.path.join(save_path, "data_status")
            os.makedirs(ds_dir, exist_ok=True)
            torch.save(data_status, os.path.join(ds_dir, f"rank{dist.get_rank()}.pt"))
            del data_status
            gc.collect()
            torch.cuda.empty_cache()

        dist.barrier()
        return

    @staticmethod
    def fsdp_save_ckpt_only(
        ckpt_dir,
        train_steps,
        model,
        logger,
        fsdp_config,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving full state checkpoint to {save_path}.")

        # Save full model on rank 0 (offload to cpu)
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            model_state_dict = model.state_dict()
            if dist.get_rank() == 0:
                torch.save(model_state_dict, os.path.join(save_path, "model.pth"))
                del model_state_dict
                gc.collect()
                torch.cuda.empty_cache()
        dist.barrier()
        return

    @staticmethod
    def try_load_ckpt(resume_from, logger, model):
        if resume_from is not None and os.path.exists(resume_from):
            logger.info(f"Loading checkpoint from {resume_from}.")
            # model_state_dict_path = os.path.join(resume_from, "model.safetensors")
            # model_state_dict = load_file(model_state_dict_path, device="cpu")
            model_state_dict = torch.load(resume_from, map_location='cpu')
            msg = model.load_state_dict(model_state_dict, strict=False)
            logger.info(msg)
            del model_state_dict
            gc.collect()
            torch.cuda.empty_cache()
        else:
            logger.info("Training from scratch.")
        return model

    @staticmethod
    def fsdp_save_fsdp_ckpt(
        ckpt_dir,
        train_steps,
        model,
        optimizer,
        scheduler,
        data_status,
        logger,
        fsdp_config,
    ):
        save_path = os.path.join(ckpt_dir, f"{train_steps:07d}")
        os.makedirs(save_path, exist_ok=True)
        logger.info(f"Saving checkpoint to {save_path}.")

        # Save sharded model using checkpoint API (each rank writes its shard under model/)
        with FSDP.state_dict_type(
            model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=True)
        ):
            model_state_dict = model.state_dict()
            model_writer = FileSystemWriter(os.path.join(save_path, "model"))
            save(model_state_dict, model_writer)
            del model_state_dict
            gc.collect()
            torch.cuda.empty_cache()

        # Save optimizer shards / local info
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            shard_index, total_shards = get_shard_info(fsdp_config)
            FSDPCheckpoint._save_optimizer_if_needed(
                fsdp_config, optimizer, save_path, shard_index, total_shards
            )

        if dist.get_rank() == 0 and scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))

        if data_status is not None:
            ds_dir = os.path.join(save_path, "data_status")
            os.makedirs(ds_dir, exist_ok=True)
            torch.save(data_status, os.path.join(ds_dir, f"rank{dist.get_rank()}.pt"))
            del data_status
            gc.collect()
            torch.cuda.empty_cache()

        dist.barrier()
        return
    
    @staticmethod
    def fsdp_save_fsdp_ckpt_to_hdfs(
        ckpt_dir,
        hdfs_ckpt_dir,
        curr_step,
        logger
    ):
        def get_all_files_including_hidden(directory):
            file_paths = []
            for root, dirs, files in os.walk(directory):
                for filename in files:
                    file_paths.append(os.path.join(root, filename))
            return file_paths
        
        local_dir = os.path.join(ckpt_dir, f"{curr_step:07d}")
        hdfs_dir = os.path.join(hdfs_ckpt_dir, f"{curr_step:07d}")
        try:
            logger.info(f"Creating HDFS directory: {hdfs_dir}")
            fs_mkdir(hdfs_dir)
            # local_files = glob.glob(os.path.join(local_dir, "**", "*"), recursive=True)
            # local_files = [f for f in local_files if os.path.isfile(f)]
            local_files = get_all_files_including_hidden(local_dir)
            if not local_files:
                logger.warning(f"No files found in {local_dir}. Nothing to upload.")
                return
            for local_file_path in local_files:
                relative_path = os.path.relpath(local_file_path, local_dir)
                hdfs_file_path = os.path.join(hdfs_dir, relative_path)
                hdfs_file_parent_dir = os.path.dirname(hdfs_file_path)
                fs_mkdir(hdfs_file_parent_dir)
                logger.debug(f"Async copying {local_file_path} to {hdfs_file_path}")
                fs_copy(local_file_path, hdfs_file_path, blocking=False)
        except Exception as e:
            logger.error(f"An error occurred during the async HDFS upload task: {e}", exc_info=True)

    @staticmethod
    def fsdp_clean_checkpoints(
        ckpt_dir,
        keep_num,
        logger,
    ):
        steps = sorted(
            [int(d) for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d))]
        )
        if len(steps) > keep_num:
            for step in steps[:-keep_num]:
                step_dir = os.path.join(ckpt_dir, f"{step:07d}")
                logger.info(f"Removing checkpoint directory: {step_dir}")
                shutil.rmtree(step_dir)
        

    @staticmethod
    def try_load_fsdp_ckpt(resume_from, model, logger=None):
        if resume_from is not None and os.path.exists(resume_from):
            if logger is not None:
                logger.info(f"Loading checkpoint from {resume_from}.")
            model_load_dir = os.path.join(resume_from, "model")

            assert isinstance(model, FSDP)
            with FSDP.state_dict_type(
                model,
                StateDictType.SHARDED_STATE_DICT,
                ShardedStateDictConfig(offload_to_cpu=True),
            ):
                model_state_dict = model.state_dict()
                model_reader = FileSystemReader(model_load_dir)
                load(model_state_dict, model_reader)
                msg = model.load_state_dict(model_state_dict, strict=False)
                if logger is not None:
                    logger.info(msg)
                del model_state_dict
                gc.collect()
                torch.cuda.empty_cache()
        else:
            if logger is not None:
                logger.info("Training from scratch.")
        return model

    @staticmethod
    def try_load_train_state(resume_from, optimizer, scheduler, fsdp_config):
        if resume_from is not None and os.path.exists(resume_from):
            shard_index, total_shards = get_shard_info(fsdp_config)

            optimizer_state_dict_path = os.path.join(
                resume_from, f"optimizer.{shard_index:05d}-of-{total_shards:05d}.pt"
            )
            if os.path.exists(optimizer_state_dict_path):
                optimizer_state_dict = torch.load(
                    optimizer_state_dict_path, map_location="cpu", weights_only=True
                )
                optimizer.load_state_dict(optimizer_state_dict)
                del optimizer_state_dict

            scheduler_state_dict_path = os.path.join(resume_from, "scheduler.pt")
            if os.path.exists(scheduler_state_dict_path) and scheduler is not None:
                scheduler_state_dict = torch.load(
                    scheduler_state_dict_path, map_location="cpu", weights_only=True
                )
                scheduler.load_state_dict(scheduler_state_dict)
                del scheduler_state_dict
            # extract path/to/001000.1f3skkcx to 1000
            train_steps = int(os.path.basename(os.path.normpath(resume_from)).split('.')[0]) + 1
            data_status_path = os.path.join(resume_from, "data_status", f"rank{dist.get_rank()}.pt")
            data_status = (
                torch.load(data_status_path, map_location="cpu", weights_only=True)
                if os.path.exists(data_status_path)
                else None
            )
        else:
            train_steps = 0
            data_status = None
        return optimizer, scheduler, train_steps, data_status

    @staticmethod
    def merge_fsdp_ckpt(resume_from, model):
        model_dir = os.path.join(resume_from, "model")
        reader = FileSystemReader(model_dir)
        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
        ):
            sharded_sd = model.state_dict()
            load(sharded_sd, reader)
            model.load_state_dict(sharded_sd, strict=False)

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            full_sd = model.state_dict()

        if dist.get_rank() == 0:
            save_file(full_sd, os.path.join(resume_from, "model.safetensors"))
        # cleanup explicit objects
        del model, full_sd, sharded_sd
        gc.collect()
        torch.cuda.empty_cache()
        return


def grad_checkpoint_check_fn(module):
    module_options = (
        Qwen3DecoderLayer,
        MLPconnector,
        TransEncoder
    )
    return isinstance(module, module_options)
