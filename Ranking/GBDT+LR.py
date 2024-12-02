import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import scipy.sparse as sp

# 假设 X_train 是训练特征，y_train 是标签
# 特征列划分
user_binary_cols = [...]  # 用户二分类特征列
user_num_cols = [...]  # 用户数值特征列
user_class_cols = [...]  # 用户多分类特征列

item_binary_cols = [...]  # 商品二分类特征列
item_num_cols = [...]  # 商品数值特征列
item_class_cols = [...]  # 商品多分类特征列


def transfer(y):
    """
    转换函数：将输入的评分转换为 0 或 1。
    如果评分 > 3，返回 1；否则返回 0。

    支持单个值或 ndarray 类型。
    """
    y = np.asarray(y)
    return np.where(y > 3, 1, 0)


y_train = transfer(y_train)


# 数据拆分
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)


# 数据预处理
def preprocess_features(
    X,
    user_binary_cols,
    user_num_cols,
    user_class_cols,
    item_binary_cols,
    item_num_cols,
    item_class_cols,
):
    """
    对特征进行预处理，包括数值特征标准化和多分类特征 One-Hot 编码。
    """
    # 提取各类特征
    user_binary = X[user_binary_cols].values
    user_num = X[user_num_cols].values
    user_class = X[user_class_cols].values

    item_binary = X[item_binary_cols].values
    item_num = X[item_num_cols].values
    item_class = X[item_class_cols].values

    # 数值特征标准化
    scaler = StandardScaler()
    user_num_scaled = scaler.fit_transform(user_num)
    item_num_scaled = scaler.fit_transform(item_num)

    # 多分类特征 One-Hot 编码
    encoder = OneHotEncoder(categories="auto", sparse=True)
    user_class_onehot = encoder.fit_transform(user_class)
    item_class_onehot = encoder.fit_transform(item_class)

    # 特征拼接
    processed_features = sp.hstack(
        (
            user_binary,
            user_num_scaled,
            user_class_onehot,
            item_binary,
            item_num_scaled,
            item_class_onehot,
        )
    )

    return processed_features, scaler, encoder


# 对训练集和验证集进行特征处理
X_train_processed, scaler, encoder = preprocess_features(
    X_train_split,
    user_binary_cols,
    user_num_cols,
    user_class_cols,
    item_binary_cols,
    item_num_cols,
    item_class_cols,
)

X_val_processed, _, _ = preprocess_features(
    X_val_split,
    user_binary_cols,
    user_num_cols,
    user_class_cols,
    item_binary_cols,
    item_num_cols,
    item_class_cols,
)

# GBDT 模型训练
gbdt_model = lgb.LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=100)
gbdt_model.fit(
    X_train_processed,
    y_train_split,
    eval_set=[(X_val_processed, y_val_split)],
    early_stopping_rounds=10,
    verbose=True,
)

# 提取 GBDT 输出的叶子节点
gbdt_train_leaf = gbdt_model.predict(X_train_processed, pred_leaf=True)
gbdt_val_leaf = gbdt_model.predict(X_val_processed, pred_leaf=True)

# 叶子节点 One-Hot 编码
leaf_encoder = OneHotEncoder(categories="auto", sparse=True)
gbdt_train_onehot = leaf_encoder.fit_transform(gbdt_train_leaf)
gbdt_val_onehot = leaf_encoder.transform(gbdt_val_leaf)

# 合并 GBDT 特征和原始数值特征
train_features = sp.hstack((gbdt_train_onehot, X_train_processed))
val_features = sp.hstack((gbdt_val_onehot, X_val_processed))

# LR 模型训练
lr_model = LogisticRegression(max_iter=100)
lr_model.fit(train_features, y_train_split)

# 模型预测
y_pred = lr_model.predict(val_features)

# 评估模型
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_val_split, y_pred)
print(f"Mean Squared Error: {mse}")


from sklearn.metrics import mean_squared_error
import numpy as np

# 预测得分
y_pred_proba = lr_model.predict_proba(val_features)[:, 1]  # 获取预测概率
y_pred_ranking = np.argsort(-y_pred_proba)  # 按预测分数降序排序

# 创建验证集用户和商品的对应关系
val_users = X_val_split["user_id"].values  # 假设用户 ID 列为 'user_id'
val_items = X_val_split["item_id"].values  # 假设商品 ID 列为 'item_id'
y_val_true = y_val_split.values  # 验证集的真实标签（如 rating 是否为高分）

# 构造结果字典 {用户: [(item, true_label, pred_score)]}
results = {}
for user, item, true_label, pred_score in zip(
    val_users, val_items, y_val_true, y_pred_proba
):
    if user not in results:
        results[user] = []
    results[user].append((item, true_label, pred_score))

# 排序每个用户的商品列表，按预测分数降序
for user in results:
    results[user] = sorted(results[user], key=lambda x: x[2], reverse=True)


# 计算 Recall@K 和 NDCG@K
def calculate_recall_and_ndcg(results, k=10):
    recall_list = []
    ndcg_list = []

    for user, items in results.items():
        # 获取实际相关的商品
        relevant_items = [
            item for item, label, _ in items if label > 3
        ]  # 假设评分 > 3 表示相关
        recommended_items = [item for item, _, _ in items[:k]]  # 推荐的 Top-K 商品

        # Recall@K
        num_relevant_and_recommended = len(set(relevant_items) & set(recommended_items))
        recall = (
            num_relevant_and_recommended / len(relevant_items)
            if len(relevant_items) > 0
            else 0
        )
        recall_list.append(recall)

        # NDCG@K
        dcg = sum(
            [
                1 / np.log2(idx + 2)
                for idx, (item, label, _) in enumerate(items[:k])
                if label > 3
            ]
        )
        idcg = sum([1 / np.log2(idx + 2) for idx in range(min(len(relevant_items), k))])
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_list.append(ndcg)

    avg_recall = np.mean(recall_list)
    avg_ndcg = np.mean(ndcg_list)

    return avg_recall, avg_ndcg


# 计算指标
recall_at_10, ndcg_at_10 = calculate_recall_and_ndcg(results, k=10)
print(f"Recall@10: {recall_at_10}")
print(f"NDCG@10: {ndcg_at_10}")

# 测试集预测
X_test_processed, _, _ = preprocess_features(
    X_test,
    user_binary_cols,
    user_num_cols,
    user_class_cols,
    item_binary_cols,
    item_num_cols,
    item_class_cols,
)

test_features = sp.hstack(
    (
        leaf_encoder.transform(gbdt_model.predict(X_test_processed, pred_leaf=True)),
        X_test_processed,
    )
)

y_test_pred_proba = lr_model.predict_proba(test_features)[:, 1]
