import numpy as np
import pandas as pd
import Preproce
from sklearn.metrics.pairwise import cosine_similarity
import json
trainSet = './data/trainSet.txt'
contentEmb = './data/embeddings.json'
tagEmb = './data/tagEmb.txt'
userEmb = './data/userEmb.txt'


# basic layer
def content_embedding(content, con_emb_dict):
    return con_emb_dict[content]
    # return [1]*768


# second layer
def average_user_tweet(user, train_df, con_emb_dict):
    user_list = []
    i = 0
    train_df = train_df.groupby(['user_id'], as_index=False).agg({'content': lambda x: list(x)})
    # train_df.to_csv(userEmb, sep='\t', index=False)
    for content_list in train_df['content'].loc[train_df['user_id'] == user]:
        for content in content_list:
            user_list += content_embedding(content, con_emb_dict)
            i += 1
    for j in range(i):
        user_list[j] /= i
    user_arr = np.array(user_list).reshape(-1, 1)
    # print(user_arr)
    return user_arr


# second layer
def average_hashtag_tweet(hashtag, train_df, con_emb_dict):
    tag_list = []
    i = 0
    train_df['hashtag'] = train_df['content'].apply(Preproce.get_hashtag)
    train_df = train_df.explode('hashtag').groupby(['hashtag'], as_index=False).agg({'content': lambda x: list(x)})
    # print(train_df.iat[1, 1])
    # train_df.to_csv(tagEmb, sep='\t', index=False)
    for content_list in train_df['content'].loc[train_df['hashtag'] == hashtag]:
        for content in content_list:
            tag_list += content_embedding(content, con_emb_dict)
            i += 1
    for j in range(i):
        tag_list[j] /= i
    tag_arr = np.array(tag_list).reshape(-1, 1)
    # print(tag_arr)
    return tag_arr


def cosine_similar(user, hashtag, train_df, con_emb_dict):
    user_arr = average_user_tweet(user, train_df, con_emb_dict)
    tag_arr = average_hashtag_tweet(hashtag, train_df, con_emb_dict)
    print(cosine_similarity(user_arr, tag_arr))
    return cosine_similarity(user_arr, tag_arr)


def rank_hashtag(train, content_emb):
    # con_emb_dict = []
    # 读取content_emb文件给con_emb_dict赋值
    train_df = pd.read_table(train)
    user_set = set(train_df['user_id'].tolist())
    train_df['hashtag'] = train_df['content'].apply(Preproce.get_hashtag)
    tag_set = set(train_df['hashtag'].explode('hashtag').tolist())
    for user in user_set:
        print('user: '+user)
        for tag in tag_set:
            print('tag: '+str(tag))
            if str(tag) != 'nan':
                print('yes')
                cosine_similar(user, tag, train_df, con_emb_dict)


if __name__ == '__main__':
    rank_hashtag(trainSet, contentEmb)
