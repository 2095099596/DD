import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau, zscore
import matplotlib.pyplot as plt
# 额外库
try:
    import dcor
except ImportError:
    print("请先安装 dcor 库：pip install dcor")
    dcor = None
try:
    from sklearn.feature_selection import mutual_info_regression
except ImportError:
    print("请先安装 scikit-learn 库：pip install scikit-learn")
    mutual_info_regression = None
# 读取Excel文件
file_path = '/mnt/hdb/LXH/data/LAI/lai1.xlsx'
df = pd.read_excel(file_path)
print(df)
# 用户自定义选择的列名（或列索引）
col1_name = 'GIMMS'
col2_name = 'ERA5213'
# 用户自定义选择的行区间（行索引，从0开始）
# start_row = 0
# end_row = 21



#
start_row = 23
end_row = 39

# 取出指定区间和列的数据
subset_norm = df.loc[start_row:end_row - 1, [col1_name, col2_name]]
# 对选中的数据进行归一化（z-score），忽略NaN
# subset_norm = subset_norm.apply(lambda x: zscore(x, nan_policy='omit'))
# 过滤掉含NaN的行，保证计算相关系数时数据有效
valid_idx = subset_norm[col1_name].notna() & subset_norm[col2_name].notna()
series1 = subset_norm.loc[valid_idx, col1_name]
series2 = subset_norm.loc[valid_idx, col2_name]
plt.figure(figsize=(8,6))
plt.scatter(series1, series2, alpha=0.7)
plt.xlabel(f'{col1_name} (z-score normalized)')
plt.ylabel(f'{col2_name} (z-score normalized)')
plt.title(f'Scatter plot of {col1_name} vs {col2_name}')
plt.grid(True)
plt.show()
if len(series1) == 0:
    print("选取的区间内没有有效数据，无法计算相关系数。")
else:
    # Pearson相关系数
    corr_pearson, p_pearson = pearsonr(series1, series2)
    print(f"Pearson相关系数: {corr_pearson:.4f}, p值: {p_pearson:.4e}")
    # Spearman相关系数
    corr_spearman, p_spearman = spearmanr(series1, series2)
    print(f"Spearman相关系数: {corr_spearman:.4f}, p值: {p_spearman:.4e}")
    # Kendall相关系数
    corr_kendall, p_kendall = kendalltau(series1, series2)
    print(f"Kendall相关系数: {corr_kendall:.4f}, p值: {p_kendall:.4e}")
    # 距离相关系数
    if dcor is not None:
        dist_corr = dcor.distance_correlation(series1, series2)
        print(f"距离相关系数: {dist_corr:.4f}")
    else:
        print("未安装 dcor 库，无法计算距离相关系数。")
    # 互信息
    if mutual_info_regression is not None:
        # mutual_info_regression要求输入为二维数组
        mi = mutual_info_regression(series1.values.reshape(-1,1), series2.values)
        print(f"互信息: {mi[0]:.4f}")
    else:
        print("未安装 scikit-learn 库，无法计算互信息。")