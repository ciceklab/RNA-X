from __future__ import annotations

import argparse
import glob
import os
from datetime import datetime
from typing import Dict, Any, List
import math


import torch, torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DistributedSampler, DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.utils.tokenizer import ResidueTokenizer
from src.data.dataset import RTDataset, RTCollator
from src.models.RT import RTHFWrapper, RTConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./hyps/bert.yaml", help="YAML with default hyper‑params")
    
    parser.add_argument("--train_presample_path", type=str, default="./chunks")
    parser.add_argument("--val_presample_path", type=str)
    parser.add_argument("--chunk_size", type=int, default=100_000)
    # parser.add_argument("--chunk_num", type=int, default=381)
    parser.add_argument("--chunk_num", type=int, default=450)

    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--resume_from_checkpoint", type=str)
    parser.add_argument("--auto_resume", default= True, action="store_true", help="Automatically resume from the latest checkpoint")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=45000)
    # parser.add_argument("--max_steps", type=int, default=37000)
    parser.add_argument("--warmup_steps", type=int, default=4800)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=int(37000//200))
    parser.add_argument("--learning_rate", type=float, default=5e-5)

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--n_layers", type=int, default=14)
    parser.add_argument("--max_position", type=int, default=4096)

    # misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="wandb")

    args, _ = parser.parse_known_args()

    with open(args.config, "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)
    cfg.update({k: v for k, v in vars(args).items() if v is not None})
    return argparse.Namespace(**cfg)

import inspect
PREFIX_CHECKPOINT_DIR = "checkpoint"
class RTTrainer(Trainer):
    def __init__(self, *args, deterministic_order=True, **kwargs):
        super().__init__(*args, **kwargs)
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            shuffle=False,
            drop_last=self.args.dataloader_drop_last,
        )
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def _save_checkpoint(self, model, trial, metrics=None):
        parent_sig = inspect.signature(super()._save_checkpoint)
        if "metrics" in parent_sig.parameters:
            maybe_path = super()._save_checkpoint(model, trial, metrics)
        else:                                          
            maybe_path = super()._save_checkpoint(model, trial)

        checkpoint_folder = maybe_path or os.path.join(
            self.args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
        )

        samples_seen = (
            self.state.global_step
            * self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.args.world_size
        )

        data_state = {
            "samples_seen": samples_seen,
            "global_step": self.state.global_step,
            "epoch": self.state.epoch,
            "world_size": self.args.world_size,
            "per_device_batch_size": self.args.per_device_train_batch_size,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "checkpoint_info": {
                "save_steps": self.args.save_steps,
                "save_total_limit": self.args.save_total_limit,
                "created_at": datetime.now().isoformat(),
            },
        }

        if self.args.process_index == 0:
            data_state_path = os.path.join(checkpoint_folder, "data_state.pt")
            torch.save(data_state, data_state_path)

            print(f"✅ Saved checkpoint: {os.path.basename(checkpoint_folder)}")
            print(f"   📁 Location: {checkpoint_folder}")
            print(f"   📊 Global step: {self.state.global_step}")
            print(f"   🔢 Samples seen: {samples_seen:,}")

            checkpoint_pattern = os.path.join(
                os.path.dirname(checkpoint_folder), f"{PREFIX_CHECKPOINT_DIR}-*"
            )
        return checkpoint_folder


def load_resume_state(resume_checkpoint_path: str) -> Dict[str, Any]:
    if not resume_checkpoint_path or not os.path.exists(resume_checkpoint_path):
        return {}
    
    data_state_path = os.path.join(resume_checkpoint_path, 'data_state.pt')
    if not os.path.exists(data_state_path):
        print(f"No data_state.pt found in {resume_checkpoint_path}")
        return {}
    
    try:
        data_state = torch.load(data_state_path, map_location='cpu')
        return data_state
    except Exception as e:
        print(f"Error loading data state: {e}")
        return {}

def list_checkpoints(output_dir: str, detailed: bool = True) -> List[str]:
    checkpoint_pattern = os.path.join(output_dir, "checkpoint-*")
    checkpoints = sorted(glob.glob(checkpoint_pattern), key=lambda x: int(x.split('-')[-1]))
    
    if not checkpoints:
        print("No checkpoints found.")
        return []
    
    print(f"📋 Found {len(checkpoints)} checkpoints in {output_dir}:")
    
    return checkpoints


def get_latest_checkpoint(output_dir: str) -> str:
    checkpoints = list_checkpoints(output_dir, detailed=False)
    if checkpoints:
        latest = checkpoints[-1]
        print(f"🔄 Latest checkpoint: {os.path.basename(latest)}")
        return latest
    return None


from accelerate import Accelerator
def main() -> None:
    args = parse_args()

    acc = Accelerator() 
    acc.print(f"🚀  rank {acc.process_index}/{acc.num_processes} - FSDP={acc.state.distributed_type}")

    
    if acc.num_processes > 1:
        rank = acc.process_index
        world_size = acc.num_processes
        print(f" rank {rank}/{world_size}")
    else:
        rank = 0
        world_size = 1
        print("Single GPU training")

    if rank == 0:
        print(f"   World size: {world_size}")
        print(f"   Local rank: {rank}")
        print(f"   Device: {acc.device}")
        
        # Additional environment debugging
        print(f"🔍 Environment check:")
        print(f"   RANK: {os.environ.get('RANK', 'Not set')}")
        print(f"   LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
        print(f"   WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
        print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        print(f"   Available GPUs: {torch.cuda.device_count()}")


    resume_from_samples = 0
    if args.auto_resume and not args.resume_from_checkpoint:
        print("Auto Resume is running")
        if os.path.exists(args.output_dir):
            args.resume_from_checkpoint = get_latest_checkpoint(args.output_dir)
            print(f"{args.resume_from_checkpoint} Checkpoint selected!")

    
    if args.resume_from_checkpoint:
        resume_state = load_resume_state(args.resume_from_checkpoint)
        if resume_state and 'samples_seen' in resume_state:
            resume_from_samples = resume_state['samples_seen']
            if rank == 0:
                print(f"   Checkpoint: {args.resume_from_checkpoint}")
                print(f"   Samples seen: {resume_from_samples:,}")
            
                
                # Show checkpoint info
                checkpoint_info = resume_state.get('checkpoint_info', {})
                if checkpoint_info:
                    created_at = checkpoint_info.get('created_at', 'Unknown')
                    print(f"   📅 Checkpoint created: {created_at}")
        else:
            if rank == 0:
                print(f"⚠️  Could not load resume state from {args.resume_from_checkpoint}")
                print("   Starting from beginning of dataset")

    for split, path in [("train", args.train_presample_path), ("val", args.val_presample_path)]:
        if not (path and os.path.isdir(path)):
            raise FileNotFoundError(f"{split} presample path invalid: {path}")
        if not glob.glob(os.path.join(path, "samples_*.pt")):
            raise FileNotFoundError(f"No samples_*.pt found in {path}")

    tokenizer = ResidueTokenizer()
    if rank == 0:
        print("Tokenizer initialized.")

    collator = RTCollator(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        eos_mask_prob=0.3,
    )
    if rank == 0:
        print("Collator initialized.")

    train_dataset = RTDataset(
        presample_path=args.train_presample_path,
        chunk_size=args.chunk_size,
        chunk_num=args.chunk_num,
        cache_size=4,
        deterministic_order=True,
        start_index=resume_from_samples,
    )

    val_dataset = RTDataset(
        presample_path=args.val_presample_path,
        chunk_size=5000,
        chunk_num=1,
        cache_size=4,
        deterministic_order=True,
        start_index=0,
    )

    if rank == 0:
        print(f"✅ Datasets created using RTDataset")
        total_samples = args.chunk_num * args.chunk_size
        remaining_samples = max(0, total_samples - resume_from_samples)
        print(f"   Total training samples: {total_samples:,}")
        print(f"   Resuming from sample: {resume_from_samples:,}")
        print(f"   Remaining samples: {remaining_samples:,}")
        print(f"   Training dataset length: {len(train_dataset):,}")
        print(f"   Validation dataset length: {len(val_dataset):,}")

    cfg = RTConfig(
        vocab_size=len(tokenizer.get_vocab()),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_position=args.max_position,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = RTHFWrapper(cfg)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if rank == 0:
        print(f"Total parameters: {count_parameters(model):,}")
        print(f"Training overview:")
        print(f"   - Total dataset size: {train_dataset.total_samples:,} samples")
        print(f"   - Remaining samples: {len(train_dataset):,}")
        print(f"   - Max Steps: {args.max_steps}")
        print(f"   - World size: {world_size}")
        print(f"   - Per-device batch size: {args.batch_size}")
        print(f"   - Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"   - Effective batch size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        if args.warmup_steps:
            print(f"   - Warmup steps: {args.warmup_steps}")

    now = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = args.output_dir if args.resume_from_checkpoint else os.path.join(args.output_dir, f"rna_model_{now}") 
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=f"RNAtranslatorX-{now}",
        
        # Data & loader settings
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        dataloader_drop_last=False,
        remove_unused_columns=False,
        
        # Training schedule
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        
        # Logging / eval / save
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        save_total_limit=200,
        load_best_model_at_end=False,
        
        # FSDP and mixed precision
        fsdp="full_shard",
        bf16=True,
        fp16=False,
        
        # Resume settings
        resume_from_checkpoint=args.resume_from_checkpoint,
        
        # Other settings
        seed=args.seed,
        report_to=args.report_to if rank == 0 else None,
        metric_for_best_model="eval_loss",
    )


    trainer = RTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        deterministic_order=True,
        # compute_metrics=compute_metrics,
        # callbacks=[TrainLogCallback()], 
    )
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # trainer.train()
    

if __name__ == "__main__":
    main()