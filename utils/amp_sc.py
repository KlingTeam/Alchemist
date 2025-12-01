import math
from typing import List, Optional, Tuple, Union

import torch


class NullCtx:
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AmpOptimizer:
    def __init__(
        self,
        mixed_precision: int,
        optimizer: torch.optim.Optimizer, names: List[str], paras: List[torch.nn.Parameter],
        grad_clip: float, n_gradient_accumulation: int = 1,
    ):
        self.enable_amp = mixed_precision > 0
        self.using_fp16_rather_bf16 = mixed_precision == 1
        
        if self.enable_amp:
            self.amp_ctx = torch.autocast('cuda', enabled=True, dtype=torch.float16 if self.using_fp16_rather_bf16 else torch.bfloat16, cache_enabled=True)
            self.scaler = torch.cuda.amp.GradScaler(init_scale=2. ** 11, growth_interval=1000) if self.using_fp16_rather_bf16 else None # only fp16 needs a scaler
        else:
            self.amp_ctx = NullCtx()
            self.scaler = None
        
        self.optimizer, self.names, self.paras = optimizer, names, paras   # paras have been filtered so everyone requires grad
        self.grad_clip = grad_clip
        self.early_clipping = self.grad_clip > 0 and not hasattr(optimizer, 'global_grad_norm')
        self.late_clipping = self.grad_clip > 0 and hasattr(optimizer, 'global_grad_norm')
        
        self.r_accu = 1 / n_gradient_accumulation   # r_accu == 1.0 / n_gradient_accumulation
    
    def backward_clip_step(
        self, stepping: bool, loss: torch.Tensor,
    ) -> Tuple[Optional[Union[torch.Tensor, float]], Optional[float]]:
        # backward
        loss = loss.mul(self.r_accu)   # r_accu == 1.0 / n_gradient_accumulation
        orig_norm = scaler_sc = None
        if self.scaler is not None:
            self.scaler.scale(loss).backward(retain_graph=False, create_graph=False)
        else:
            loss.backward(retain_graph=False, create_graph=False)
        
        if stepping:
            if self.scaler is not None: self.scaler.unscale_(self.optimizer)

            ## TODO: 收集最后一层的梯度向量 (在clip之前)
            # 获取最后一层参数的梯度
            last_layer_grads = []
            # 方法1: 假设self.paras是列表,最后几个是最后一层
            # 通常最后一层包含weight和bias两个参数
            for param in list(self.paras)[-2:]:  # 取最后两个参数(通常是weight和bias)
                if param.grad is not None:
                    last_layer_grads.append(param.grad.detach().clone().view(-1))
            
            if last_layer_grads:
                grad_vector = torch.cat(last_layer_grads).cpu().numpy()
                if hasattr(self, 'gradient_collector'):
                    self.gradient_collector.gradients.append(grad_vector)
            ## TODO END
            if self.early_clipping:
                orig_norm = torch.nn.utils.clip_grad_norm_(self.paras, self.grad_clip)
            
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                scaler_sc: float = self.scaler.get_scale()
                if scaler_sc > 32768.: # fp16 will overflow when >65536, so multiply 32768 could be dangerous
                    self.scaler.update(new_scale=32768.)
                else:
                    self.scaler.update()
                try:
                    scaler_sc = float(math.log2(max(scaler_sc,1e-10)))
                except Exception as e:
                    print(f'[scaler_sc = {scaler_sc}]\n' * 15, flush=True)
                    raise e
            else:
                self.optimizer.step()
            
            if self.late_clipping:
                orig_norm = self.optimizer.global_grad_norm
            
            self.optimizer.zero_grad(set_to_none=True)
        
        return orig_norm, scaler_sc
    
    def state_dict(self):
        return {
            'optimizer': self.optimizer.state_dict()
        } if self.scaler is None else {
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
    
    def load_state_dict(self, state, strict=True):
        if self.scaler is not None:
            try: self.scaler.load_state_dict(state['scaler'])
            except Exception as e: print(f'[fp16 load_state_dict err] {e}')
        self.optimizer.load_state_dict(state['optimizer'])

        
## TODO: 添加GradientCollector类定义
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class GradientCollector:
    def __init__(self):
        self.gradients = []
        self.labels = []
    
    def add_label(self, label):
        """单独添加标签(如果需要的话)"""
        self.labels.append(label)
    
    def plot_tsne(self, perplexity=30, n_iter=1000, save_path='gradient_tsne.png', use_pca=True, pca_components=50):
        """绘制t-SNE图"""
        if len(self.gradients) < 2:
            print(f"需要至少2个梯度向量,当前只有{len(self.gradients)}个")
            return
        
        # 转换为numpy数组
        grad_matrix = np.array(self.gradients)
        print(f"最后一层梯度矩阵形状: {grad_matrix.shape}")
        
        # 可选: 先用PCA降维
        if use_pca and grad_matrix.shape[1] > pca_components:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(pca_components, grad_matrix.shape[0]-1, grad_matrix.shape[1]))
            grad_matrix = pca.fit_transform(grad_matrix)
            print(f"PCA降维到 {grad_matrix.shape[1]} 维, 解释方差比: {pca.explained_variance_ratio_.sum():.4f}")
        
        # 执行t-SNE
        print("开始执行t-SNE...")
        actual_perplexity = min(perplexity, len(self.gradients)-1, 50)
        tsne = TSNE(n_components=2, perplexity=actual_perplexity, 
                    n_iter=n_iter, random_state=42, verbose=1)
        embeddings = tsne.fit_transform(grad_matrix)
        print("t-SNE完成!")
        
        # 绘图
        plt.figure(figsize=(12, 10))
        
        if self.labels and len(self.labels) == len(self.gradients):
            # 如果有标签,用不同颜色表示
            unique_labels = sorted(list(set(self.labels)))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for idx, label in enumerate(unique_labels):
                indices = [i for i, l in enumerate(self.labels) if l == label]
                plt.scatter(embeddings[indices, 0], embeddings[indices, 1], 
                           c=[colors[idx]], label=f'Class {label}', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        else:
            # 没有标签,用渐变色表示样本顺序
            scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], 
                       c=range(len(embeddings)), cmap='viridis', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter, label='Sample Index')
        
        plt.title(f't-SNE Visualization of Last Layer Gradients (n={len(self.gradients)})', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
        plt.close()
    
    def clear(self):
        """清空收集的梯度"""
        self.gradients = []
        self.labels = []
    
    def save_gradients(self, save_path='gradients.npz'):
        """保存梯度数据"""
        np.savez(save_path, gradients=np.array(self.gradients), 
                 labels=np.array(self.labels) if self.labels else None)
        print(f"梯度数据已保存到: {save_path}")
    
    def load_gradients(self, load_path='gradients.npz'):
        """加载梯度数据"""
        data = np.load(load_path)
        self.gradients = data['gradients'].tolist()
        if data['labels'] is not None:
            self.labels = data['labels'].tolist()
        print(f"已加载 {len(self.gradients)} 个梯度向量")
## TODO END