import pandas as pd
import numpy as np
import time as t
import Preproce
np.random.seed(int(t.time()))
train_path = './data/trainWeibo.txt'
test_path = './data/testWeibo.txt'
freq_tag = './data/sortTagWeibo.txt'
train_df = pd.read_table(train_path)
test_df = pd.read_table(test_path)
freq_tag_df = pd.read_table(freq_tag)


def sort_tag(train):
    df = train
    df['topics'] = train['content'].apply(Preproce.get_hashtag)
    df = df.explode('topics').groupby(['topics'], as_index=False)['user_id'].agg({'cnt': 'count'}).sort_values(by=['cnt'], ascending=False)
    print(df)
    df.to_csv('./data/sortTagWeibo.txt', sep='\t', index=False)


def random_rec(dataframe):
    tag_rec = []
    for i in range(5):
        tag_rec.append(dataframe['topics'].loc[np.random.randint(100)])
    return tag_rec


def popular_rec(dataframe):
    tag_rec = []
    for i in range(5):
        tag_rec.append(dataframe['topics'].loc[i])
    return tag_rec


def latest_rec(user, dataframe):
    tag_rec = []
    # dataframe = dataframe.sort_values(by=['crawl_time'], ascending=False)
    user_df = dataframe['content'].loc[dataframe['user_id'] == user]
    for i in range(5):
        tag_rec += Preproce.get_hashtag(user_df.iloc[i])

    print(list(set(tag_rec)))
    return list(set(tag_rec))


def eval_rec(dataframe1, dataframe2, dataframe3):
    user_train_df = dataframe1.drop(['_id', 'crawl_time', 'weibo_url', 'like_num', 'repost_num', 'comment_num', 'image_url', 'content', 'topics'], axis=1)
    user_set = set(list(user_train_df['user_id']))
    print(len(user_set))
    user_test_df = dataframe2.drop(['_id', 'crawl_time', 'weibo_url', 'like_num', 'repost_num', 'comment_num', 'image_url', 'topics'], axis=1)

    success = 0
    for user in user_set:                         # 重复出现的user要记得筛掉
        # tag_list = random_rec(dataframe3)
        # tag_list = popular_rec(dataframe3)
        dataframe1 = dataframe1.sort_values(by=['crawl_time'], ascending=False) # for latest_rec only
        tag_list = latest_rec(user, dataframe1)
        for tag in tag_list:
            if tag in Preproce.get_hashtag(user_test_df[user_test_df['user_id'] == user]['content']):
                success += 1
                break

    # print("random recommendation: "+str(success/len(user_set)))
    # print("popularity recommendation: " + str(success / len(user_set)))
    print("latest recommendation: " + str(success / len(user_set)))


if __name__ == '__main__':
    eval_rec(train_df, test_df, freq_tag_df)
    # sort_tag(train_df)
