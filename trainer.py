import json
import math
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_scheduler

from dataset import DataCollator, NERDataset
from model import build_model
from plot import plot_training_history
from utils import dataset_path, load_config, set_seed

_run = None


def start_run(config):
    global _run
    if not config.get("use_swanlab", True):
        return
    import swanlab
    _run = swanlab.init(
        project=config.get("swanlab_project", "qwen2.5-ner"),
        experiment_name=config.get("swanlab_run_name", "custom-trainer"),
        mode=config.get("swanlab_mode", "cloud"),
        config=config,
    )


def log_metrics(values, step):
    if _run is None:
        return
    import swanlab
    swanlab.log(values, step=step)


def finish_run():
    if _run is None:
        return
    import swanlab
    swanlab.finish()


class NERTrainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        collator = DataCollator(tokenizer.pad_token_id)
        workers = config.get("dataloader_num_workers", 0)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["per_device_train_batch_size"],
            shuffle=True,
            collate_fn=collator,
            num_workers=workers,
            pin_memory=True,
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            batch_size=config["per_device_eval_batch_size"],
            shuffle=False,
            collate_fn=collator,
            num_workers=workers,
            pin_memory=True,
        )

        parameters = [param for param in model.parameters() if param.requires_grad]
        self.device = parameters[0].device
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.0),
        )

        accumulation = config["gradient_accumulation_steps"]
        steps_per_epoch = math.ceil(len(self.train_loader) / accumulation)
        self.total_steps = steps_per_epoch * int(config["num_train_epochs"])
        warmup_steps = int(self.total_steps * config.get("warmup_ratio", 0.0))
        self.scheduler = get_scheduler(
            config.get("lr_scheduler_type", "cosine"),
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )
        self.history = {"train_loss": [], "eval_loss": []}

    def move_batch(self, batch):
        return {name: tensor.to(self.device, non_blocking=True) for name, tensor in batch.items()}

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        for batch in self.eval_loader:
            batch = self.move_batch(batch)
            total_loss += self.model(**batch).loss.float().item()
        self.model.train()
        return total_loss / len(self.eval_loader)

    def save_checkpoint(self, step):
        checkpoint = self.output_dir / f"checkpoint-{step}"
        self.model.save_pretrained(checkpoint)
        self.tokenizer.save_pretrained(checkpoint)

        limit = self.config.get("save_total_limit", 2)
        checkpoints = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda path: int(path.name.split("-")[-1]),
        )
        for old_checkpoint in checkpoints[:-limit]:
            shutil.rmtree(old_checkpoint)

    def train(self):
        config = self.config
        accumulation = config["gradient_accumulation_steps"]
        logging_steps = config.get("logging_steps", 10)
        eval_steps = config.get("eval_steps", 500)
        save_steps = config.get("save_steps", 500)
        max_grad_norm = config.get("max_grad_norm", 1.0)
        global_step = 0
        running_loss = 0.0
        running_batches = 0

        start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        with (self.output_dir / "training_config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        start_run(config)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(int(config["num_train_epochs"])):
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
            for batch_index, batch in enumerate(progress):
                batch = self.move_batch(batch)
                group_start = batch_index - batch_index % accumulation
                group_size = min(accumulation, len(self.train_loader) - group_start)

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda" and config.get("bf16", True),
                ):
                    loss = self.model(**batch).loss

                (loss / group_size).backward()
                running_loss += loss.detach().float().item()
                running_batches += 1

                last_batch = batch_index + 1 == len(self.train_loader)
                if (batch_index + 1) % accumulation != 0 and not last_batch:
                    continue

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % logging_steps == 0:
                    train_loss = running_loss / running_batches
                    learning_rate = self.scheduler.get_last_lr()[0]
                    self.history["train_loss"].append((global_step, train_loss))
                    log_metrics({"train/loss": train_loss, "train/lr": learning_rate}, global_step)
                    progress.set_postfix(loss=f"{train_loss:.4f}")
                    running_loss = 0.0
                    running_batches = 0

                if global_step % eval_steps == 0:
                    eval_loss = self.evaluate()
                    self.history["eval_loss"].append((global_step, eval_loss))
                    log_metrics({"eval/loss": eval_loss}, global_step)

                if global_step % save_steps == 0:
                    self.save_checkpoint(global_step)

            if not self.history["eval_loss"] or self.history["eval_loss"][-1][0] != global_step:
                eval_loss = self.evaluate()
                self.history["eval_loss"].append((global_step, eval_loss))
                log_metrics({"eval/loss": eval_loss}, global_step)

        if running_batches:
            train_loss = running_loss / running_batches
            self.history["train_loss"].append((global_step, train_loss))
            log_metrics({"train/loss": train_loss}, global_step)

        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        with (self.output_dir / "training_history.json").open("w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        plot_training_history(self.history, self.output_dir)

        peak_memory = 0.0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3
            log_metrics({"system/peak_memory_gb": peak_memory}, global_step)
            print(f"Peak GPU memory: {peak_memory:.2f} GB")
        summary = {
            "total_steps": global_step,
            "train_seconds": time.time() - start_time,
            "peak_memory_gb": peak_memory,
        }
        with (self.output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        finish_run()


def train(config_file):
    config = load_config(config_file)
    set_seed(config.get("seed", 42))
    model, tokenizer = build_model(config)
    train_dataset = NERDataset(
        dataset_path(config["dataset"]),
        tokenizer,
        max_length=config["cutoff_len"],
        max_samples=config.get("max_samples"),
    )
    eval_dataset = NERDataset(
        dataset_path(config["eval_dataset"]),
        tokenizer,
        max_length=config["cutoff_len"],
    )
    trainer = NERTrainer(model, tokenizer, train_dataset, eval_dataset, config)
    trainer.train()
