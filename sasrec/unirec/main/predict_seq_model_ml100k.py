import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import ndcg_score
import seaborn as sns
from typing import Dict, List, Tuple

class SaturationAnalyzer:
    def __init__(self, data_path: str, model, config: Dict):
        """初始化分析器
        Args:
            data_path: 数据路径
            model: 训练好的模型
            config: 配置信息
        """
        self.data_path = data_path
        self.model = model
        self.config = config
        
        # 加载饱和度数据
        self.user_saturation = pd.read_pickle(f"{data_path}/saturation_analysis/user_saturation_vectors.pkl")
        self.item_saturation = pd.read_pickle(f"{data_path}/saturation_analysis/item_saturation_vectors.pkl")
        
        # 加载测试数据
        self.test_data = pd.read_csv(f"{data_path}/train.csv", sep='\t')
        
    def calculate_interaction_saturation(self, user_id: int, item_id: int) -> float:
        """计算用户-物品对的饱和度得分"""
        user_vec = self.user_saturation.loc[user_id].values
        item_vec = self.item_saturation.loc[str(item_id)].values
        return np.mean(user_vec * item_vec)  # 使用平均值作为综合得分
        
    def analyze_predictions(self, 
                          top_k: List[int] = [10, 20], 
                          n_groups: int = 5) -> Tuple[Dict, pd.DataFrame]:
        """分析预测结果与饱和度的关系
        Args:
            top_k: 评估的k值列表
            n_groups: 饱和度分组数量
        Returns:
            group_metrics: 各组的评估指标
            detailed_results: 详细的结果DataFrame
        """
        self.model.eval()
        results = []
        
        # 对每个用户进行评估
        for user_id in self.test_data['user_id'].unique():
            user_items = self.test_data[self.test_data['user_id'] == user_id]
            
            # 获取所有候选物品的预测分数
            all_items = np.arange(1, self.config['n_items'])
            user_tensor = torch.LongTensor([user_id] * len(all_items))
            item_tensor = torch.LongTensor(all_items)
            
            with torch.no_grad():
                scores = self.model(user_tensor, item_tensor)
                scores = scores.cpu().numpy()
            
            # 计算真实标签
            true_items = user_items['item_id'].values
            y_true = np.zeros_like(scores)
            y_true[true_items - 1] = 1  # -1因为item_id从1开始
            
            # 计算NDCG@k
            for k in top_k:
                ndcg = ndcg_score([y_true], [scores], k=k)
                
                # 计算平均饱和度
                saturation_scores = []
                for item_id in true_items:
                    sat_score = self.calculate_interaction_saturation(user_id, item_id)
                    saturation_scores.append(sat_score)
                avg_saturation = np.mean(saturation_scores)
                
                results.append({
                    'user_id': user_id,
                    'saturation': avg_saturation,
                    f'ndcg@{k}': ndcg
                })
        
        # 转换为DataFrame并进行分组分析
        results_df = pd.DataFrame(results)
        results_df['saturation_group'] = pd.qcut(results_df['saturation'], q=n_groups, labels=[f'G{i+1}' for i in range(n_groups)])
        
        # 计算每个组的平均指标
        group_metrics = {}
        for k in top_k:
            group_ndcg = results_df.groupby('saturation_group')[f'ndcg@{k}'].agg(['mean', 'std']).round(4)
            group_metrics[f'ndcg@{k}'] = group_ndcg
            
        return group_metrics, results_df
    
    def plot_saturation_ndcg_relationship(self, results_df: pd.DataFrame, output_path: str):
        """可视化饱和度与NDCG的关系"""
        plt.figure(figsize=(12, 6))
        
        # 绘制散点图和趋势线
        for col in [c for c in results_df.columns if 'ndcg@' in c]:
            sns.regplot(data=results_df, x='saturation', y=col, label=col)
            
        plt.xlabel('Saturation Score')
        plt.ylabel('NDCG')
        plt.title('Relationship between Data Saturation and NDCG')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_path}/saturation_ndcg_relationship.png")
        plt.close()
        
        # 绘制箱形图
        plt.figure(figsize=(12, 6))
        results_df_melted = pd.melt(results_df, 
                                  id_vars=['saturation_group'], 
                                  value_vars=[c for c in results_df.columns if 'ndcg@' in c],
                                  var_name='metric', 
                                  value_name='value')
        
        sns.boxplot(data=results_df_melted, x='saturation_group', y='value', hue='metric')
        plt.xlabel('Saturation Group')
        plt.ylabel('NDCG')
        plt.title('NDCG Distribution across Saturation Groups')
        plt.savefig(f"{output_path}/saturation_groups_ndcg.png")
        plt.close()

def run_saturation_analysis(model_path: str, data_path: str, output_path: str):
    """运行饱和度分析实验"""
    # 加载模型
    model, config = general.load_model_freely(model_path)
    
    # 创建分析器
    analyzer = SaturationAnalyzer(data_path, model, config)
    
    # 进行分析
    group_metrics, results_df = analyzer.analyze_predictions(top_k=[10, 20], n_groups=5)
    
    # 保存结果
    results_df.to_csv(f"{output_path}/saturation_analysis_results.csv", index=False)
    
    # 生成可视化
    analyzer.plot_saturation_ndcg_relationship(results_df, output_path)
    
    # 打印分组结果
    print("\nGroup-wise NDCG metrics:")
    for k, metrics in group_metrics.items():
        print(f"\n{k}:")
        print(metrics)

if __name__ == "__main__":
    # 在模型训练完成后调用
    model_path = "/home/xieao1/UniRec/output/ml-100k/SASRec/train/checkpoint_2025-01-11_095800_60/SASRec-SASRec-ml-100k.pth"
    data_path = "/home/xieao1/UniRec/data/ml-100k"
    output_path = f"/home/xieao1/UniRec/output/ml-100k/SASRec/xxx/analysis"

    run_saturation_analysis(model_path, data_path, output_path)