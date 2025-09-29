# This script provides an example of using FSA on Llama3-8B.
import os
import random

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (AutoTokenizer, LlamaForCausalLM,
                          get_linear_schedule_with_warmup)


def set_all_seeds(seed=42):
    """Comprehensive seeding for reproducibility"""

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For distributed training
    os.environ['PYTHONHASHSEED'] = str(seed)

    # For some CUDA operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # Enable deterministic algorithms
    torch.use_deterministic_algorithms(True, warn_only=True)


class SparseLlamaAttention(nn.Module):
    """Replace standard Llama attention with NSA"""

    def __init__(self, config, layer_idx=None, args=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', self.num_heads)

        # Initialize FSA/NSA
        if args.attn_mode == "FSA":
            from fsa.module.fsa import FlashSparseAttention, RopeConfig
            sparse_cls = FlashSparseAttention
        else:
            from nsa_ref.module import NativeSparseAttention, RopeConfig
            sparse_cls = NativeSparseAttention

        self.sparse_attn = sparse_cls(
            hidden_size=self.hidden_size,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_key_value_heads,
            head_dim=128,
            kernel_size=32,
            kernel_stride=getattr(args, 'kernel_stride', 16),
            block_size=getattr(args, 'block_size', 128),
            topk=getattr(args, 'topk', 64),
            init_blocks=1,
            local_blocks=2,
            window_size=512,
            rope_config=RopeConfig(
                max_position_embeddings=131072,
                head_dim=128,
                rope_theta=500000,
                rope_scaling={
                    "factor": 8.0,
                    "high_freq_factor": 4.0,
                    "low_freq_factor": 1.0,
                    "original_max_position_embeddings": 8192,
                    "rope_type": "llama3",
                },
            ),
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        bsz, seq_len, _ = hidden_states.shape

        # Create cumulative sequence lengths for NSA
        cu_seqlens = torch.arange(
            0, (bsz + 1) * seq_len, seq_len,
            device=hidden_states.device,
            dtype=torch.int32
        )
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
            packed_hidden_states = hidden_states.view(-1, hidden_states.size(-1))
            attn_output = self.sparse_attn(packed_hidden_states, cu_seqlens)
            # Reshape back to (batch_size, seq_len, hidden_size)
            attn_output = attn_output.view(bsz, seq_len, -1)

            return attn_output, None


def reinitialize_attention_params(model):
    """Reinitialize all attention parameters with Xavier uniform"""
    for layer in model.model.layers:
        attn = layer.self_attn
        # Reinitialize all parameters in the Llama attention module
        for param in attn.parameters():
            torch.nn.init.xavier_uniform_(param)
    return model


def replace_llama_attention(model, args, accelerator):
    """Replace all Llama attention layers with NSA and reinitialize"""

    if args.attn_mode != "FA":
        for layer_idx, layer in enumerate(model.model.layers):
            # Replace the self_attn module
            layer.self_attn = SparseLlamaAttention(
                model.config,
                layer_idx=layer_idx,
                args=args
            ).to(torch.bfloat16)
            # Replace attention layers with NSA
        if accelerator.is_main_process:
            print(f"Replacing attention layers with {args.attn_mode}...")
    else:
        # Reinitialize Full Attention parameters
        reinitialize_attention_params(model)
        if accelerator.is_main_process:
            print("Reinitialized Llama attention parameters...")
    return model


class ArxivPapersDataset(Dataset):
    """ArxivPapers dataset for training"""

    def __init__(self, tokenizer, num_samples=1000, max_length=512):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = min(max_length, 16384)  # Keep under 16K limit
        self.data = self._load_data()

    def _load_data(self):
        """Load and preprocess ArxivPapers data"""
        # Load ArxivPapers dataset
        dataset = load_dataset("CShorten/ML-ArXiv-Papers", split="train")

        texts = []
        for item in dataset:
            text = item['abstract'].strip()

            # Filter for substantial content (longer than wikitext)
            if len(text) > 1000 and len(text.split()) > 200:
                texts.append(text)

                if len(texts) >= self.num_samples:
                    break

        print(f"Loaded {len(texts)} ArxivPapers samples")
        avg_words = sum(len(t.split()) for t in texts) // len(texts)
        print(f"Average length: ~{avg_words} words")
        return texts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        mask = encoding["attention_mask"].squeeze()

        return {
            "input_ids": encoding["input_ids"].squeeze()[mask],
            "labels": encoding["input_ids"].squeeze()[mask]  # For causal LM
        }


def collate_fn(batch):
    """Custom collate function"""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch])
    }


def train_nsa_llama():
    """Main training function with Accelerate"""

    # Initialize arguments
    args = Args()

    assert args.batch_size == 1, "This example script only allows one sequence."

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision='bf16',
        log_with=None,
        project_dir=args.output_dir,
    )

    # Set seed
    set_all_seeds()

    # Print info on main process
    if accelerator.is_main_process:
        print("Setup: ")
        print(f"Starting NSA Llama training (attn mode: {args.attn_mode}, seqlen: {args.max_length})")
        print(f"Mixed precision: {accelerator.mixed_precision}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base Llama model
    if accelerator.is_main_process:
        print("Loading base Llama-3-8B model...")

    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        torch_dtype=torch.bfloat16,
        device_map=None,  # Let accelerator handle device placement
    )

    model = replace_llama_attention(model, args, accelerator)
    model.gradient_checkpointing_enable()

    # Wait for all processes
    accelerator.wait_for_everyone()

    # Create datasets
    train_dataset = ArxivPapersDataset(
        tokenizer,
        num_samples=args.num_train_samples,
        max_length=args.max_length
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Calculate total training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=max_train_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        print(f"Total training steps: {max_train_steps}")
        print(f"Steps per epoch: {num_update_steps_per_epoch}")

    # Training loop
    model.train()
    global_step = 0
    total_loss = 0

    for epoch in range(args.num_epochs):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        epoch_loss = 0
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=not accelerator.is_local_main_process
        )

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                # Forward pass

                outputs = model(**batch)

                # Compute loss (causal LM loss)
                labels = batch["labels"]
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # Backward pass
                accelerator.backward(loss)

                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Update metrics
                total_loss += loss.detach().float()
                epoch_loss += loss.detach().float()

                if accelerator.sync_gradients:
                    global_step += 1

                # Update progress bar
                if accelerator.is_local_main_process:
                    progress_bar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'lr': f"{lr_scheduler.get_last_lr()[0]:.2e}",
                        'step': global_step
                    })

                # Logging
                if global_step % 1 == 0 and accelerator.is_main_process:
                    print(f"Step {global_step}: Loss = {loss.item():.4f}", flush=True)

        # End of epoch logging
        epoch_avg_loss = epoch_loss / len(train_dataloader)
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1} Average Loss: {epoch_avg_loss:.4f}")

    if accelerator.is_main_process:
        print("Training completed!")


class Args:
    """Training arguments"""
    def __init__(self):
        # NSA specific args
        self.hidden_size = 4096
        self.q_heads = 32
        self.kv_heads = 8
        self.kernel_stride = 16
        self.block_size = 64
        self.topk = 16
        self.attn_mode = "NSA"  # choose in ["FSA", "NSA", "FA"]

        # Training args
        self.output_dir = "./nsa_llama_training"
        self.num_epochs = 3
        # Batch size must be 1 to use sequence packing
        self.batch_size = 1
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.01
        self.warmup_steps = 100
        # This script only shows an example for single GPU fine-tuning; therefore, the seqlen is only set at 8K
        self.max_length = 8192
        self.num_train_samples = 1000
        self.num_eval_samples = 200


if __name__ == "__main__":
    # Start training
    train_nsa_llama()
