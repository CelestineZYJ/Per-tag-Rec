import numpy as np
import pandas as pd
import Preproce
from sklearn.metrics.pairwise import cosine_similarity
import json
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
tag_list = list(set(train_df['hashtag'].explode('hashtag').tolist()))[1:200]
# 读取content_emb文件给con_emb_dict赋值
con_emb_dict = []


# basic layer
def content_embedding(content):
    # return con_emb_dict[content]
    return [1]*768


# second layer
def average_user_tweet(user_list):
    user_arr_list = []
    user_emb_list = []
    i = 0
    # train_df.to_csv(userEmb, sep='\t', index=False)
    for user in user_list:
        print('arr calculate for user: '+user)
        for content_list in content_user_df['content'].loc[content_user_df['user_id'] == user]:
            for content in content_list:
                user_emb_list += content_embedding(content)
                i += 1
        for j in range(i):
            user_emb_list[j] /= i
        user_arr_list.append(np.array(user_emb_list).reshape(-1, 1))
    # print(user_arr)
    user_arr_dict = dict(zip(user_list, user_arr_list))
    print(user_arr_dict)
    return user_arr_dict


# second layer
def average_hashtag_tweet(tag_list):
    tag_arr_list = []
    tag_emb_list = []
    i = 0
    # print(train_df.iat[1, 1])
    # train_df.to_csv(tagEmb, sep='\t', index=False)
    for tag in tag_list:
        print('arr calculate for tag: ' + tag)
        for content_list in content_tag_df['content'].loc[content_tag_df['hashtag'] == tag]:
            for content in content_list:
                tag_emb_list += content_embedding(content)
                i += 1
        for j in range(i):
            tag_emb_list[j] /= i
        tag_arr_list.append(np.array(tag_emb_list).reshape(-1, 1))
    tag_arr_dict = dict(zip(tag_list, tag_arr_list))
    print(tag_arr_dict)
    return tag_arr_dict


def cosine_similar(user, hashtag, user_arr_dict, tag_arr_dict):
    # print(cosine_similarity(user_arr, tag_arr))
    return cosine_similarity(user_arr_dict[user], tag_arr_dict[hashtag])


def rank_hashtag():
    # dictionary to return hashtag recommendation score to all user
    spe_user_cos_list = []

    user_arr_dict = average_user_tweet(user_list)
    tag_arr_dict = average_hashtag_tweet(tag_list)

    for user in user_list:
        cosine_list = []
        print('user: '+user)
        for tag in tag_list:
            print('tag: '+str(tag))
            if str(tag) != 'nan':
                # print('yes')
                cosine_list.append(cosine_similar(user, tag, user_arr_dict, tag_arr_dict))
        tag_cos_dict = dict(zip(tag_list, cosine_list))
        tag_cos_dict = sorted(tag_cos_dict.items())
        # print(tag_cos_dict)
        spe_user_cos_list.append(tag_cos_dict)
        print(spe_user_cos_list)
        print(len(spe_user_cos_list))

    rank_dict = dict(zip(user_list, spe_user_cos_list))
    return rank_dict


def embedding_rec(user, rank_dict):
    return ['#NODAYSOFF']                            #rank_dict[user][0]


def eval_rec(user_lis, tag_lis):
    print(len(user_lis))
    print(len(tag_lis))
    # calculate the whole rec dict of cosine_similarity of each hashtag to each user
    rank_dict = rank_hashtag()
    user_test_df = test_df.drop(['tweet_id', 'time', 'hashtag'], axis=1)

    success = 0
    for user in user_lis:                         # 重复出现的user要记得筛掉
        print('score of tag rec for: '+user)
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
