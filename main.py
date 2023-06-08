import random
import pandas as pd
import numpy as np


# 现在相似度矩阵的计算效率比较低, 之后可以尝试一下使用并行进行优化

def cosine_similarity(v1, v2):
    """
    :param v1: 向量1
    :param v2: 向量2
    :return: 两个向量的余弦相似度
    """
    # 计算向量的点积
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    return dot_product / (norm_v1 * norm_v2)


def get_similarity_matrix(rating_mat):
    height, width = rating_mat.shape
    similarity = np.zeros((height, height))
    for i in range(height):
        for j in range(i + 1):
            if i == j:
                similarity[i][j] = 1.0
            else:
                similarity[i][j] = cosine_similarity(rating_mat[i], rating_mat[j])
                similarity[j][i] = similarity[i][j]
    return similarity


def recommend(user, similarity, rating, h_split_index, N=10):
    """
    :param rating: 用户的评分矩阵
    :param user: 用户的序号
    :param similarity: 用户的相似度矩阵
    :param h_split_index: 垂直切割点
    :param N: 推荐的数量
    :return itemList: 给用户推荐的N个item的列表, 其中每个元素表示一个item的序号
    """
    max_indices = np.argsort(similarity[user])[:-1][-N:]
    max_elements = similarity[user][max_indices]
    print(f'与用户{user}最相似的{N}个用户为:{max_indices}, 对应的相似度为{max_elements}')
    total_rating = np.zeros(rating.shape[1])
    # 计算相似度较高的用户的总评分的均值
    cnt = 0
    for i in max_indices:
        if i == user:
            continue
        total_rating = np.add(total_rating, rating[i])
        cnt += 1
    avg_rating = total_rating / cnt
    avg_rating = avg_rating[h_split_index:]
    item_rating_rank = np.argsort(avg_rating)[-N:] + h_split_index
    print(f'给用户{user}推荐的商品有:{item_rating_rank}')
    return item_rating_rank


if __name__ == '__main__':
    hdr = ['uid', 'user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv('train.dat', delimiter=",", header=None, names=hdr)
    mat = data.values[:, 1:4]  # 表中内容: 用户 物品 评分
    n_users = data.user_id.unique().shape[0]
    n_items = data.item_id.unique().shape[0]
    print(f"user number: {n_users}, item number: {n_items}")
    # 计算用户评分矩阵
    rating = np.zeros((n_users, np.max(mat[:, 1])))
    for line in mat:
        rating[line[0] - 1, line[1] - 1] = line[2]
    # 按列截取十分之一的用户作为验证集
    split_ratio = 0.1
    split_index = int(rating.shape[0] * split_ratio)
    test_data, train_data = np.split(rating, [split_index])  # 训练数据和测试数据
    # 对于验证集中的用户, 截取30%的列评分数据作为预测的部分
    h_split_ratio = 0.7
    h_split_index = int(rating.shape[1] * h_split_ratio)
    left_data, right_data = np.hsplit(rating, [split_index])  # 左侧的部分为完全已知的部分, 可以用来计算用户相似度
    # 计算用户相似度矩阵
    user_similarity = get_similarity_matrix(left_data)
    # 随机取一些用户
    user_list = random.sample(range(0, split_index), 10)
    # 选10个测试集中的用户, 每个用户根据相似度最高的10个已知用户, 给自己推荐10个自己未知部分的产品, 和实际rating里面自己评分过的取一个交集, 就是命中的物品
    for user in user_list:
        rec_list = recommend(user, user_similarity, rating, h_split_index)
        rating_exist = set()
        for i in range(h_split_index, rating.shape[1]):
            if rating[user][i] != 0:
                rating_exist.add(i)
        hit = len(set(rec_list) & rating_exist)
        precision = 1.0 * hit / 10
        recall = 1.0 * hit / (rating.shape[1]-h_split_index)
