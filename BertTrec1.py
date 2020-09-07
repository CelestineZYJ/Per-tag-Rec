import numpy as np
import pandas as pd
import Preproce
from sklearn.metrics.pairwise import cosine_similarity
import json
import pdb

trainSet = './data/trainSet.txt'
testSet = './data/testSet.txt'
contentEmb = './data/embeddings.json'
tagEmb = './data/tagEmb.txt'
userEmb = './data/userEmb.txt'
train_df = pd.read_table(trainSet)
test_df = pd.read_table(testSet)
train_df['hashtag'] = train_df['content'].apply(Preproce.get_hashtag)
content_user_df = train_df.groupby(['user_id'], as_index=False).agg({'content': lambda x: list(x)})
content_tag_df = train_df.explode('hashtag').groupby(['hashtag'], as_index=False).agg({'content': lambda x: list(x)})
user_list = list(set(train_df['user_id'].tolist()))[1:200]
tag_list = list(set(train_df['hashtag'].explode('hashtag').tolist()))
tag_list.remove('nan')
# 读取content_emb文件给con_emb_dict赋值

with open("data/embeddings.json", "r") as f:
    con_emb_dict = json.load(f)


# basic layer
def content_embedding(content):
    try:
        return con_emb_dict[content]
    except:
        return [0] * 768


# second layer
def average_user_tweet(user_list):
    user_arr_dict = {}
    for user in user_list:
        embed_list = []
        content_list = content_user_df['content'].loc[content_user_df['user_id'] == user].tolist()[0]
        for content in content_list:
            embed_list.append(content_embedding(content))
        embed_list = np.mean(np.array(embed_list), axis=0) # (768, )
        user_arr_dict[user] = embed_list

    return user_arr_dict


# second layer
def average_hashtag_tweet(tag_list):
    tag_arr_dict = {}
    for tag in tag_list:
        embed_list = []
        try:
            content_list = content_tag_df['content'].loc[content_tag_df['hashtag'] == tag].tolist()[0]
        except:
            pdb.set_trace()
        for content in content_list:
            embed_list.append(content_embedding(content))
        embed_list = np.mean(np.array(embed_list), axis=0) # (768, )
        tag_arr_dict[tag] = embed_list

    return tag_arr_dict


def cosine_similar(user, hashtag, user_arr_dict, tag_arr_dict):
    return cosine_similarity(user_arr_dict[user].reshape(1, -1), tag_arr_dict[hashtag].reshape(1, -1))


def rank_hashtag():
    # dictionary to return hashtag recommendation score to all user
    spe_user_cos_list = []

    user_arr_dict = average_user_tweet(user_list)
    tag_arr_dict = average_hashtag_tweet(tag_list)

    for user in user_list:
        cosine_list = []
        for tag in tag_list:
            if str(tag) != 'nan':
                # print('yes')
                cosine_list.append(cosine_similar(user, tag, user_arr_dict, tag_arr_dict))
        tag_cos_dict = dict(zip(tag_list, cosine_list))
        tag_cos_dict = sorted(tag_cos_dict.items())
        # print(tag_cos_dict)
        spe_user_cos_list.append(tag_cos_dict)

    rank_dict = dict(zip(user_list, spe_user_cos_list))
    return rank_dict


def embedding_rec(user, rank_dict):
    t1, t2 = rank_dict[user][0]
    if isinstance(t1, str):
        return [t1]
    else:
        return [t2]
    # return ['#NODAYSOFF']                            #rank_dict[user][0]


def eval_rec(user_lis, tag_lis):
    # calculate the whole rec dict of cosine_similarity of each hashtag to each user
    rank_dict = rank_hashtag()
    user_test_df = test_df.drop(['tweet_id', 'time', 'hashtag'], axis=1)

    success = 0
    for user in user_lis:                         # 重复出现的user要记得筛掉
        # print('score of tag rec for: '+user)
        tag_lis = embedding_rec(user, rank_dict)
        for tag in tag_lis:
            if tag in Preproce.get_hashtag(user_test_df[user_test_df['user_id'] == user]['content']):
                success += 1
                break

    print("embedding recommendation: " + str(success / len(user_list)))


if __name__ == '__main__':
    eval_rec(user_list, tag_list)
    '''
    a = ['user1', 'user2', 'user3']
    b = ['tag1', 'tag2', 'tag3', 'tag4']
    c = ['1', '2', '3', '4']
    d1 = dict(zip(b, c))
    print(d1)
    d2 = dict(zip(b, c))
    d3 = dict(zip(b, c))
    d = [d1, d1, d3]
    e = dict(zip(a, d))
    print(e)
    '''
