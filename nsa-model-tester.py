import torch
from transformers import BertTokenizer
import jieba
from rouge_chinese import Rouge
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import logging
from typing import List, Dict
import json

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NSAModelTester:
    def __init__(
        self,
        model_path: str,
        tokenizer: BertTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.tokenizer = tokenizer
        self.model = self._load_model(model_path)
        self.rouge = Rouge()
        
    def _load_model(self, model_path: str):
        """加载预训练模型"""
        try:
            # 从app41.py中导入必要的类
            from app41 import NSAConfig, NSAModel
            
            # 初始化配置
            config = NSAConfig(
                vocab_size=self.tokenizer.vocab_size,
                max_seq_length=512,
                hidden_size=768,
                num_attention_heads=12,
                num_hidden_layers=6,
                compress_ratio=4,
                select_k=16,
                window_size=64
            )
            
            # 创建模型实例
            model = NSAModel(config)
            
            # 加载预训练权重
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model = model.to(self.device)
            model.eval()
            
            logger.info("模型加载成功")
            return model
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
            
    def generate_text(
        self,
        prompt: str,
        max_length: int = 50,
        temperature: float = 0.6,
        top_k: int = 30,
        top_p: float = 0.9,
        batch_size: int = 1
    ) -> str:
        """生成文本"""
        try:
            # 编码输入文本
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding="max_length",
                add_special_tokens=True
            )
            
            # 移动到指定设备
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            #logger.info(f"Input shape: {input_ids.shape}")
            
            # 初始化输出序列
            output_ids = input_ids.clone()
            current_attention_mask = attention_mask.clone()
            
            # 自回归生成
            with torch.no_grad():
                for _ in range(max_length):
                    # 获取模型预测
                    logits = self.model(
                        input_ids=output_ids,
                        attention_mask=current_attention_mask
                    )
                    
                    # 获取最后一个token的预测
                    next_token_logits = logits[:, -1, :] / temperature
                    
                    # Top-K 过滤
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Top-P 过滤
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # 采样下一个token
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # 检查是否生成了终止标记
                    if next_token.item() in [self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                        break
                    
                    # 将新token添加到序列中
                    output_ids = torch.cat([output_ids, next_token], dim=1)
                    # 扩展attention mask
                    current_attention_mask = torch.cat([
                        current_attention_mask,
                        torch.ones((batch_size, 1), device=self.device)
                    ], dim=1)
            
            # 解码生成的文本
            generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return generated_text[len(prompt):]  # 只返回新生成的部分
            
        except Exception as e:
            logger.error(f"文本生成失败: {str(e)}")
            return ""
            
    def evaluate_quality(
        self,
        generated_texts: List[str],
        reference_texts: List[str]
    ) -> Dict:
        """评估生成文本的质量"""
        try:
            # 准备评估指标
            scores = {
                "rouge-1": [],
                "rouge-2": [],
                "rouge-l": [],
                "bleu-4": []
            }
            
            for gen, ref in zip(generated_texts, reference_texts):
                # 分词
                gen_tokens = " ".join(jieba.cut(gen))
                ref_tokens = " ".join(jieba.cut(ref))
                
                # 计算ROUGE分数
                rouge_scores = self.rouge.get_scores(gen_tokens, ref_tokens)[0]
                scores["rouge-1"].append(rouge_scores["rouge-1"]["f"])
                scores["rouge-2"].append(rouge_scores["rouge-2"]["f"])
                scores["rouge-l"].append(rouge_scores["rouge-l"]["f"])
                
                # 计算BLEU分数
                ref_tokens = [list(jieba.cut(ref))]
                gen_tokens = list(jieba.cut(gen))
                bleu_score = sentence_bleu(ref_tokens, gen_tokens)
                scores["bleu-4"].append(bleu_score)
            
            # 计算平均分数
            final_scores = {
                metric: np.mean(values) for metric, values in scores.items()
            }
            
            return final_scores
            
        except Exception as e:
            logger.error(f"质量评估失败: {str(e)}")
            return {}
            
    def run_test_suite(
        self,
        test_prompts: List[str],
        reference_texts: List[str] = None,
        verbose: bool = True
    ):
        """运行完整的测试套件"""
        try:
            generated_texts = []
            
            # 生成文本
            for prompt in test_prompts:
                generated = self.generate_text(prompt)
                generated_texts.append(generated)
                
                if verbose:
                    print(f"\n输入提示: {prompt}")
                    print(f"生成文本: {generated}")
            
            # 如果有参考文本，进行质量评估
            """
            if reference_texts:
                quality_scores = self.evaluate_quality(generated_texts, reference_texts)
                
                if verbose:
                    print("\n质量评估结果:")
                    for metric, score in quality_scores.items():
                        print(f"{metric}: {score:.4f}")
                        
                return generated_texts, quality_scores
            """
            return generated_texts, None
            
        except Exception as e:
            logger.error(f"测试套件运行失败: {str(e)}")
            return [], None

def main():
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 初始化测试器
    tester = NSAModelTester(
        model_path="nsa_chinese_model.pth",
        tokenizer=tokenizer
    )
    
    # 准备测试样例
    test_prompts = [
        "夏天的海邊",
        "這是一個平和的冬天",
        "關於科技創新",
        "面對面對挑戰，內心充滿了喜悅，不禁想到過去的經歷。"
    ]
    
    # 参考答案（用于评估）
    reference_texts = [
        "陽光明媚，微風輕拂，是個適合出門散步的好日子。",
        "看到了技術進步帶來的巨大變革，但也要警惕其潛在風險。",
        "深厚的歷史積澱，需要我們繼續傳承和發揚。",
        "人工智能將與人類共同創造更美好的生活。"
    ]
    
    # 运行测试
    generated_texts, quality_scores = tester.run_test_suite(
        test_prompts,
        reference_texts,
        verbose=True
    )
    
    # 保存测试结果
    
    results = {
        "test_cases": [
            {
                "prompt": prompt,
                "generated": generated
            }
            for prompt, generated in zip(test_prompts, generated_texts)
        ],
        "quality_scores": quality_scores
    }
    
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
