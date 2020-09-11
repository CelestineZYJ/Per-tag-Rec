import numpy as np
import pandas as pd
import Preproce
from sklearn.metrics.pairwise import cosine_similarity
import json
import pdb
from tqdm import tqdm

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
user_tag_df = train_df.explode('hashtag').groupby(['user_id'], as_index=False).agg({'hashtag': lambda x: list(x)})
user_list = list(set(train_df['user_id'].tolist())) # [0:2000]
Tag_df = train_df.explode('hashtag')
Tag_list = list(set(Tag_df['hashtag'].tolist()))[1:]
# Tag_list.remove('nan')
# 读取content_emb文件给con_emb_dict赋值user_tag_lis

with open("data/embeddings.json", "r") as f:
    con_emb_dict = json.load(f)


# basic layer
def content_embedding(content):
    try:
        return con_emb_dict[content]
    except:
        return [0] * 768


# second layer
def average_user_tweet(user_lis):
    user_arr_dict = {}
    for user in user_lis:
        embed_list = []
        content_list = content_user_df['content'].loc[content_user_df['user_id'] == user].tolist()[0]
        for content in content_list:
            embed_list.append(content_embedding(content))
        embed_list = np.mean(np.array(embed_list), axis=0) # (768, )
        user_arr_dict[user] = embed_list

    return user_arr_dict


# second layer
def average_hashtag_tweet(tag_lis):
    tag_arr_dict = {}
    for tag in tag_lis:
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
    return float(cosine_similarity(user_arr_dict[user].reshape(1, -1), tag_arr_dict[hashtag].reshape(1, -1)))###############################


def rank_hashtag():
    # dictionary to return hashtag recommendation score to all user
    spe_user_cos_list = []

    user_arr_dict = average_user_tweet(user_list)
    tag_arr_dict = average_hashtag_tweet(Tag_list)

    for user in tqdm(user_list):
        cosine_list = []
        spec_tag_lis = user_tag_df['hashtag'].loc[user_tag_df['user_id'] == user].tolist()[0]
        print(str(user)+': '+str(len(spec_tag_lis)))     ##################################################
        for tag in spec_tag_lis:
            if str(tag) != 'nan':
                # print('yes')
                cosine_list.append(cosine_similar(user, tag, user_arr_dict, tag_arr_dict))
        # tag_cos_dict = OrderedDict()
        tag_cos_dict = dict(zip(cosine_list, spec_tag_lis)) ########################################################
        tag_cos_dict = sorted(tag_cos_dict.items(), reverse=True)####################################################
        # print(tag_cos_dict)
        spe_user_cos_list.append(tag_cos_dict)

    rank_dict = dict(zip(user_list, spe_user_cos_list))
    return rank_dict


def embedding_rec(user, rank_dict):
    '''
    t1, t2 = rank_dict[user][0][1]
    if isinstance(t1, str):
        return [t1]
    else:
        return [t2]
    '''
    tag_lis = []
    for i in range(5):
        try:
            tag_lis.append(rank_dict[user][i][1])
        except:
            tag_lis.append('None')
    print(tag_lis)
    # rank_dict[user][0][1]返回tuple(cosine_sim, hashtag)中的hashtag, cosine_sim是float，hashtag是str
    return tag_lis
    # return str(rank_dict[user][0][1])


def eval_rec(user_lis):
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
    eval_rec(user_list)
    '''
    a = ['user1', 'user2', 'user3']
    b = ['tag1', 'tag2', 'tag3', 'tag4']
    c = ['3', '2', '1', '4']
    d1 = OrderedDict()
    d1 = dict(zip(b, c))
    print(d1)
    '''

    '''
    d2 = dict(zip(b, c))
    d3 = dict(zip(b, c))
    d = [d1, d1, d3]
    e = dict(zip(a, d))
    print(e)
    '''

