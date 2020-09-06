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


# basic layer
def content_embedding(content, con_emb_dict):
    # return con_emb_dict[content]
    return [1]*768


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
    # print(cosine_similarity(user_arr, tag_arr))
    return cosine_similarity(user_arr, tag_arr)


def rank_hashtag(train_df, con_emb_dict):
    user_set = set(train_df['user_id'].tolist())
    train_df['hashtag'] = train_df['content'].apply(Preproce.get_hashtag)
    tag_set = set(train_df['hashtag'].explode('hashtag').tolist())

    # dictionary to return hashtag recommendation score to all user
    user_list = list(user_set)
    spe_user_cos_list = []
    for user in user_list:
        tag_list = list(tag_set)[1:4]
        cosine_list = []
        print('user: '+user)
        for tag in tag_list: # tag_set:
            print('tag: '+str(tag))
            if str(tag) != 'nan':
                # print('yes')
                cosine_list.append(cosine_similar(user, tag, train_df, con_emb_dict))
        tag_cos_dict = dict(zip(tag_list, cosine_list))
        tag_cos_dict = sorted(tag_cos_dict.items())
        # print(tag_cos_dict)
        spe_user_cos_list.append(tag_cos_dict)
        print(spe_user_cos_list)
        # print(len(spe_user_cos_list))

    rank_dict = dict(zip(user_list, spe_user_cos_list))
    return rank_dict


def embedding_rec(user, rank_dict):
    return ['#NODAYSOFF']                            #rank_dict[user][0]


def eval_rec(train, test, content_emb):
    # 读取content_emb文件给con_emb_dict赋值
    con_emb_dict = []
    
    train_df = pd.read_table(train)
    test_df = pd.read_table(test)

    # calculate the whole rec dict of cosine_similarity of each hashtag to each user
    rank_dict = rank_hashtag(train_df, con_emb_dict)

    user_train_df = train_df.drop(['tweet_id', 'time', 'content', 'hashtag'], axis=1)
    user_set = set(list(user_train_df['user_id']))
    print(len(user_set))
    user_test_df = test_df.drop(['tweet_id', 'time', 'hashtag'], axis=1)

    success = 0
    for user in user_set:                         # 重复出现的user要记得筛掉
        print('score of tag rec for: '+user)
        tag_list = embedding_rec(user, rank_dict)
        for tag in tag_list:
            if tag in Preproce.get_hashtag(user_test_df[user_test_df['user_id'] == user]['content']):
                success += 1
                break

    print("embedding recommendation: " + str(success / len(user_set)))


if __name__ == '__main__':
    eval_rec(trainSet, testSet, contentEmb)
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
