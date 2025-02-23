import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import json
import numpy as np
from typing import Optional
import math
import random
from chinese_data_generator02 import ChineseDataGenerator
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler
import gc
import os
import time, datetime


# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 設置 CUDA 記憶體分配器配置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

class TrainingMetrics:
    def __init__(self):
        self.train_losses = []  # 每個batch的loss
        self.epoch_losses = []  # 每個epoch的平均loss
        self.validation_losses = []  # 每次驗證的loss
        self.learning_rates = []  # 學習率追蹤
        self.gpu_memory_usage = []  # GPU記憶體使用追蹤
        self.best_loss = float('inf')  # 最佳loss
        self.no_improvement_count = 0  # 用於early stopping
        
    def update_batch_loss(self, loss):
        self.train_losses.append(loss)
        
    def update_epoch_loss(self, loss):
        self.epoch_losses.append(loss)
        
    def update_validation_loss(self, loss):
        self.validation_losses.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
            self.no_improvement_count = 0
            return True
        else:
            self.no_improvement_count += 1
            return False
            
    def should_early_stop(self, patience=3):
        return self.no_improvement_count >= patience

class OptimizedTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 8,
        mixed_precision: bool = True,
        early_stopping_patience: int = 3,
        logging_steps: int = 10
    ):
        # 檢測可用的GPU並計算可用VRAM
        self.n_gpu = torch.cuda.device_count()
        self.devices = []
        total_memory = 0
        
        for i in range(self.n_gpu):
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            if i == 0 or memory >= (torch.cuda.get_device_properties(0).total_memory / 1024**3) * 0.75:
                self.devices.append(i)
                total_memory += memory
                
        self.n_gpu = len(self.devices)
        self.main_device = "cuda:0"
        
        # 調整batch size和梯度累積步數
        self.adjusted_batch_size = self._calculate_safe_batch_size(batch_size, total_memory)
        self.gradient_accumulation_steps = self._calculate_accumulation_steps(
            batch_size, 
            self.adjusted_batch_size
        )
        
        print(f"Using {self.n_gpu} GPUs: {self.devices}")
        print(f"Original batch size: {batch_size}")
        print(f"Adjusted batch size: {self.adjusted_batch_size}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        
        # 清理GPU記憶體
        self._clean_gpu_memory()
        
        # 移動模型到主GPU並啟用記憶體優化
        self.model = model.to(self.main_device)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.devices)
        
        # 配置數據加載器，使用較小的prefetch factor
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.adjusted_batch_size,
            shuffle=True,
            num_workers=2,  # 減少worker數量
            pin_memory=True,
            prefetch_factor=2,  # 限制預取量
            persistent_workers=True  # 保持worker進程
        )
        
        if test_dataset:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.adjusted_batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
        else:
            self.test_loader = None
            
        # 調整學習率
        adjusted_lr = learning_rate * math.sqrt(
            self.adjusted_batch_size * self.gradient_accumulation_steps / batch_size
        )
        
        # 配置優化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=adjusted_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # 配置學習率調度器
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,  # 每5個epoch重啟
            T_mult=2,
            eta_min=adjusted_lr * 0.01
        )
        
        # 混合精度訓練設置
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None
        
        # 訓練參數
        self.num_epochs = num_epochs
        
        # 顯示GPU信息
        self._print_gpu_info()

        # 添加新的訓練指標追蹤
        self.metrics = TrainingMetrics()
        self.early_stopping_patience = early_stopping_patience
        self.logging_steps = logging_steps
            
    def _calculate_safe_batch_size(self, original_batch_size, total_memory):
        """計算安全的batch size"""
        # 假設每個樣本約需要0.5GB VRAM，再加上模型和優化器的開銷
        safe_batch_size = int((total_memory * 0.3) / 0.5)  # 只使用30%的VRAM給batch
        safe_batch_size = min(original_batch_size, safe_batch_size)
        safe_batch_size = max(1, (safe_batch_size // 8) * 8)  # 確保是8的倍數
        return safe_batch_size
        
    def _calculate_accumulation_steps(self, target_batch_size, actual_batch_size):
        """計算需要的梯度累積步數"""
        steps = math.ceil(target_batch_size / actual_batch_size)
        return max(steps, 8)  # 至少8步
        
    def _clean_gpu_memory(self):
        """清理GPU記憶體"""
        torch.cuda.empty_cache()
        gc.collect()
        
    def _print_gpu_info(self):
        """打印GPU信息"""
        for i in self.devices:
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {total:.1f}GB total, "
                  f"{allocated:.1f}GB allocated, "
                  f"{cached:.1f}GB cached")
            
    def _check_memory(self):
        """監控GPU記憶體使用"""
        for i in self.devices:
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            free = (torch.cuda.get_device_properties(i).total_memory / 1024**3) - allocated
            print(f"GPU {i} - Allocated: {allocated:.2f}GB, "
                  f"Cached: {cached:.2f}GB, "
                  f"Free: {free:.2f}GB")
            
    @torch.cuda.amp.autocast()
    def _forward_pass(self, batch):
        """執行前向傳播"""
        try:
            input_ids = batch["input_ids"].to(self.main_device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.main_device, non_blocking=True)
            labels = batch["labels"].to(self.main_device, non_blocking=True)
            
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._clean_gpu_memory()
                raise e
            raise e
    def _log_training_info(self, epoch, batch_idx, loss, lr):
        """記錄詳細的訓練資訊"""
        # 計算進度百分比
        progress = (batch_idx + 1) / len(self.train_loader) * 100
        
        # 計算預估剩餘時間
        batch_time = time.time() - self.last_log_time
        remaining_batches = len(self.train_loader) - (batch_idx + 1)
        eta = datetime.timedelta(seconds=int(batch_time * remaining_batches))
        
        # 更新時間戳
        self.last_log_time = time.time()
        
        # 獲取GPU記憶體使用情況
        gpu_memory = {i: torch.cuda.memory_allocated(i) / 1024**3 for i in self.devices}
        
        # 格式化輸出
        log_str = (
            f"Epoch: {epoch+1}/{self.num_epochs} | "
            f"Progress: {progress:.2f}% | "
            f"Batch: {batch_idx+1}/{len(self.train_loader)} | "
            f"Loss: {loss:.12f} | "
            f"LR: {lr:.12f} | "
            f"ETA: {eta} | "
            f"GPU Memory: {gpu_memory}"
        )
        
        # 如果有驗證集，添加最佳驗證loss
        if self.test_loader:
            log_str += f" | Best Val Loss: {self.metrics.best_loss:.6f}"
        
        logger.info(log_str)
        
        # 更新指標
        self.metrics.update_batch_loss(loss)
        self.metrics.learning_rates.append(lr)
        self.metrics.gpu_memory_usage.append(gpu_memory)    


    def train(self):
        """優化後的訓練循環，包含更詳細的訓練資訊"""
        self.model.train()
        self._clean_gpu_memory()
        self.last_log_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            total_loss = 0
            num_batches = 0
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(self.train_loader):
                try:
                    # 使用混合精度訓練
                    if self.mixed_precision:
                        with autocast():
                            loss = self._forward_pass(batch)
                            loss = loss / self.gradient_accumulation_steps
                            
                        self.scaler.scale(loss).backward()
                        
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                            self.scheduler.step()
                    else:
                        loss = self._forward_pass(batch)
                        loss = loss / self.gradient_accumulation_steps
                        loss.backward()
                        
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            self.scheduler.step()
                    
                    total_loss += loss.item() * self.gradient_accumulation_steps
                    num_batches += 1
                    
                    # 定期記錄訓練資訊
                    if (batch_idx + 1) % self.logging_steps == 0:
                        avg_loss = total_loss / num_batches
                        lr = self.scheduler.get_last_lr()[0]
                        self._log_training_info(epoch, batch_idx, avg_loss, lr)
                        self._check_memory()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning("WARNING: out of memory, attempting recovery...")
                        self._clean_gpu_memory()
                        
                        # 減少batch size並重新配置
                        self.adjusted_batch_size = max(1, self.adjusted_batch_size // 2)
                        self.gradient_accumulation_steps *= 2
                        logger.info(f"Reducing batch size to {self.adjusted_batch_size}")
                        logger.info(f"Increasing accumulation steps to {self.gradient_accumulation_steps}")
                        
                        # 重新創建數據加載器
                        self.train_loader = DataLoader(
                            self.train_loader.dataset,
                            batch_size=self.adjusted_batch_size,
                            shuffle=True,
                            num_workers=2,
                            pin_memory=True,
                            prefetch_factor=2,
                            persistent_workers=True
                        )
                        
                        continue
                    else:
                        raise e
            # 計算並記錄epoch統計資訊
            epoch_avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start_time
            
            logger.info(
                f"\nEpoch {epoch+1} Summary:\n"
                f"Average Loss: {epoch_avg_loss:.6f}\n"
                f"Time Elapsed: {datetime.timedelta(seconds=int(epoch_time))}\n"
                f"Learning Rate: {self.scheduler.get_last_lr()[0]:.8f}"
            )
            
            # 更新epoch指標
            self.metrics.update_epoch_loss(epoch_avg_loss)
            
            # 評估並檢查early stopping
            if self.test_loader:
                val_loss = self.evaluate()
                improved = self.metrics.update_validation_loss(val_loss)
                
                if improved:
                    logger.info("New best validation loss! Saving model checkpoint...")
                    self.save_model("best_model_checkpoint.pth")
                
                if self.metrics.should_early_stop(self.early_stopping_patience):
                    logger.info(
                        f"Early stopping triggered after {self.early_stopping_patience} "
                        "epochs without improvement"
                    )
                    break

            # 評估
            if self.test_loader:
                self.evaluate()
                
    def evaluate(self):
        """增強的評估函數"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        eval_start_time = time.time()
        
        with torch.no_grad():
            for batch in self.test_loader:
                if self.mixed_precision:
                    with autocast():
                        loss = self._forward_pass(batch)
                else:
                    loss = self._forward_pass(batch)
                    
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches
        eval_time = time.time() - eval_start_time
        
        logger.info(
            f"\nValidation Results:\n"
            f"Average Loss: {avg_loss:.6f}\n"
            f"Time Elapsed: {datetime.timedelta(seconds=int(eval_time))}"
        )
        
        return avg_loss
        
    def save_model(self, path: str):
        """保存模型"""
        if isinstance(self.model, nn.DataParallel):
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)
            
    def load_model(self, path: str):
        """加載模型"""
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(torch.load(path))
        else:
            self.model.load_state_dict(torch.load(path))

class NSAConfig:
    def __init__(
        self,
        vocab_size: int = 21128,  # BERT中文詞表大小
        max_seq_length: int = 512,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 6,
        compress_ratio: int = 4,
        select_k: int = 16,
        window_size: int = 64,
    ):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.compress_ratio = compress_ratio
        self.select_k = select_k
        self.window_size = window_size

class NSAAttentionExtended(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.head_size
        self.scale = 1.0 / math.sqrt(self.head_size)
        
        # Query, Key, Value projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Output projection
        self.output = nn.Linear(self.hidden_size * 3, self.hidden_size)
        
        # Branch gates
        self.branch_gate = nn.Linear(self.hidden_size, 3)
        
        # Dropouts
        self.attention_dropout = nn.Dropout(0.1)
        self.output_dropout = nn.Dropout(0.1)
        
        # Additional parameters
        self.compress_ratio = config.compress_ratio
        self.select_k = config.select_k
        self.window_size = config.window_size
        
        # Compression layer
        self.compress = nn.Linear(config.hidden_size * config.compress_ratio, config.hidden_size)
        
        # Selection layer
        self.selection_score = nn.Linear(self.hidden_size, 1)

    def _adjust_tensor_size(self, tensor, target_size, dim=1):
        """调整张量大小以匹配目标大小"""
        current_size = tensor.size(dim)
        if current_size == target_size:
            return tensor
            
        if current_size < target_size:
            # 需要填充
            pad_size = target_size - current_size
            padding = torch.zeros_like(tensor.narrow(dim, 0, 1)).repeat_interleave(pad_size, dim=dim)
            return torch.cat([tensor, padding], dim=dim)
        else:
            # 需要裁剪
            return tensor.narrow(dim, 0, target_size)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def compress_attention(self, hidden_states, attention_mask=None):
        """压缩注意力机制"""
        batch_size, seq_length, _ = hidden_states.size()
        
        # 调整序列长度为compress_ratio的倍数
        pad_length = (self.compress_ratio - seq_length % self.compress_ratio) % self.compress_ratio
        if pad_length > 0:
            hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_length))
            if attention_mask is not None:
                attention_mask = nn.functional.pad(attention_mask, (0, pad_length))
        
        # 重塑为blocks
        new_seq_length = hidden_states.size(1)
        blocks = hidden_states.view(batch_size, -1, self.compress_ratio, self.hidden_size)
        blocks = blocks.reshape(batch_size, -1, self.compress_ratio * self.hidden_size)
        
        # 压缩blocks
        compressed = self.compress(blocks)
        
        # 计算自注意力
        query = self.query(compressed)
        key = self.key(compressed)
        value = self.value(compressed)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        context = torch.matmul(attention_probs, value)
        
        # 调整输出大小以匹配原始序列长度
        context = self._adjust_tensor_size(context, seq_length)
        
        return context

    def select_attention(self, hidden_states, attention_mask=None):
        """选择性注意力机制"""
        batch_size, seq_length, _ = hidden_states.size()
        
        # 计算有效的select_k
        effective_k = min(self.select_k, seq_length)
        
        # 计算选择分数
        scores = self.selection_score(hidden_states).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.bool(), float('-inf'))
        
        # 选择top-k位置
        _, indices = torch.topk(scores, k=effective_k, dim=-1)
        indices = indices.sort(dim=-1)[0]
        
        # 收集选定的状态
        selected = torch.gather(
            hidden_states,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        )
        
        # 计算自注意力
        query = self.query(selected)
        key = self.key(selected)
        value = self.value(selected)
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        context = torch.matmul(attention_probs, value)
        
        # 调整输出大小以匹配原始序列长度
        context = self._adjust_tensor_size(context, seq_length)
        
        return context

    def window_attention(self, hidden_states, attention_mask=None):
        """滑动窗口注意力机制"""
        batch_size, seq_length, _ = hidden_states.size()
        
        # 计算有效窗口大小
        effective_window = min(self.window_size, seq_length)
        
        # 添加填充
        pad_length = (effective_window - seq_length % effective_window) % effective_window
        if pad_length > 0:
            hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_length))
            if attention_mask is not None:
                attention_mask = nn.functional.pad(attention_mask, (0, pad_length))
        
        # 创建滑动窗口
        windows = []
        for i in range(0, hidden_states.size(1) - effective_window + 1, effective_window // 2):
            windows.append(hidden_states[:, i:i + effective_window])
        
        # 处理每个窗口
        window_outputs = []
        for window in windows:
            # 计算自注意力
            query = self.query(window)
            key = self.key(window)
            value = self.value(window)
            
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = self.attention_dropout(attention_probs)
            
            context = torch.matmul(attention_probs, value)
            window_outputs.append(context)
        
        # 合并窗口输出
        output = torch.cat(window_outputs, dim=1)
        
        # 调整输出大小以匹配原始序列长度
        output = self._adjust_tensor_size(output, seq_length)
        
        return output

    def forward(self, hidden_states, attention_mask=None):
        """前向传播"""
        batch_size, seq_length, _ = hidden_states.size()
        
        # 计算分支权重
        gates = self.branch_gate(hidden_states)
        gates = torch.sigmoid(gates)
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)
        
        # 分别处理每个分支
        compress_output = self.compress_attention(hidden_states, attention_mask)
        select_output = self.select_attention(hidden_states, attention_mask)
        window_output = self.window_attention(hidden_states, attention_mask)
        
        # 确保所有输出具有相同的大小
        compress_output = self._adjust_tensor_size(compress_output, seq_length)
        select_output = self._adjust_tensor_size(select_output, seq_length)
        window_output = self._adjust_tensor_size(window_output, seq_length)
        
        # 应用门控
        gated_compress = compress_output * gates[..., 0:1]
        gated_select = select_output * gates[..., 1:2]
        gated_window = window_output * gates[..., 2:3]
        
        # 合并输出
        combined = torch.cat([gated_compress, gated_select, gated_window], dim=-1)
        
        # 最终输出
        output = self.output(combined)
        output = self.output_dropout(output)
        
        # 残差连接和归一化
        output = output * 0.5 + hidden_states * 0.5
        output = nn.functional.layer_norm(
            output,
            [output.size(-1)],
            weight=None,
            bias=None,
            eps=1e-6
        )
        
        return output

class NSABlockExtended(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = NSAAttentionExtended(config)
        self.intermediate = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.output = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.layernorm1 = nn.LayerNorm(config.hidden_size)
        self.layernorm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()
        
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layernorm1(hidden_states + attention_output)
        
        intermediate_output = self.intermediate(hidden_states)
        intermediate_output = self.activation(intermediate_output)
        
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        
        output = self.layernorm2(hidden_states + layer_output)
        return output

class NSAModel(nn.Module):
    def __init__(self, config: NSAConfig):
        super().__init__()
        self.config = config
        
        # Keep original embedding structure
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_seq_length,
            config.hidden_size
        )
        
        # Initialize embeddings with small values
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        
        # Keep original layer structure
        self.layers = nn.ModuleList(
            [NSABlockExtended(config) for _ in range(config.num_hidden_layers)]
        )
        
        # Keep original normalization and head
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize lm_head
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lm_head.bias)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Get embeddings with gradient clipping
        hidden_states = self.embeddings(input_ids)
        hidden_states = torch.clamp(hidden_states, min=-100, max=100)
        
        # Add position embeddings
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        position_embeddings = torch.clamp(position_embeddings, min=-100, max=100)
        
        # Combine embeddings
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        
        # Process through transformer layers with stability improvements
        for layer in self.layers:
            layer_output = layer(hidden_states, attention_mask)
            # Add scaled residual connection
            hidden_states = hidden_states * 0.5 + layer_output * 0.5
            # Clip values to prevent explosion
            hidden_states = torch.clamp(hidden_states, min=-100, max=100)
        
        # Generate logits with careful scaling
        prediction_scores = self.lm_head(hidden_states)
        prediction_scores = prediction_scores / math.sqrt(self.config.hidden_size)
        prediction_scores = torch.clamp(prediction_scores, min=-100, max=100)
        
        # Calculate loss if labels are provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            return loss
            
        return prediction_scores

class ChineseTextDataset(Dataset):
    def __init__(
        self,
        data,  # 可以是文件路徑或數據列表
        tokenizer: BertTokenizer,
        max_length: int = 512
    ):
        if isinstance(data, str):
            # 如果是文件路徑，從文件讀取數據
            with open(data, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            # 如果是數據列表，直接使用
            self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> dict:
        text = self.data[idx]["text"]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.test_loader = None
        if test_dataset:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )
        self.num_epochs = num_epochs
        self.device = device
        
    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.num_epochs}, Batch {batch_idx+1}/{len(self.train_loader)}, Loss: {loss.item():.12f}")
                
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Average Loss: {avg_loss:.12f}")
            
            if self.test_loader:
                self.evaluate()
                
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.test_loader)
        print(f"Validation Loss: {avg_loss:.12f}")
        
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))

def main():
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 生成訓練數據
    data_generator = ChineseDataGenerator()
    dataset = data_generator.generate_dataset(20000)
    
    # 設定模型配置
    #config = NSAConfig(
    #    vocab_size=tokenizer.vocab_size,
    #   max_seq_length=512,
    #    hidden_size=768,
    #    num_attention_heads=12,
    #    num_hidden_layers=6,
    #    compress_ratio=4,
    #    select_k=16,
    #    window_size=64
    #)
    config = NSAConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_length=512,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        compress_ratio=4,
        select_k=32,
        window_size=64
    )
    # 創建模型
    model = NSAModel(config)
    
    # 準備數據集
    train_size = int(0.9 * len(dataset))
    indices = torch.randperm(len(dataset))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_data = [dataset[i] for i in train_indices]
    test_data = [dataset[i] for i in test_indices]
    
    train_dataset = ChineseTextDataset(train_data, tokenizer)
    test_dataset = ChineseTextDataset(test_data, tokenizer)
    
    # 使用優化後的訓練器
    trainer = OptimizedTrainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=16,
        learning_rate=1e-5,
        num_epochs=30,
        gradient_accumulation_steps=4,
        mixed_precision=True
    )
    
    # 開始訓練
    trainer.train()
    
    # 保存模型
    trainer.save_model("nsa_chinese_model.pth")    

if __name__ == "__main__":
    main()