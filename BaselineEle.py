import Preproce
import pandas as pd
import numpy as np
import time as t
np.random.seed(int(t.time()))

trainEle4 = './data/trainEle.txt'
testEle5 = './data/testEle.txt'
sortedTag = './data/sortTagEle.txt'


def random_rec(df):
    tag_rec = []
    for i in range(5):
        tag_rec.append(df['hashtag'].loc[np.random.randint(100)])
    return tag_rec


def popular_rec(df):
    tag_rec = []
    for i in range(5):
        tag_rec.append(df['hashtag'].loc[i])
    return tag_rec


def latest_rec(user, df):
    tag_rec = []
    df = df.sort_values(by=['tweet_id'], ascending=False)
    user_df = df['content'].loc[df['user_id'] == user]
    for i in range(5):
        tag_rec += Preproce.get_hashtag(user_df.iloc[i])

    return list(set(tag_rec))


def eval_rec(trainF, testF, sortF):
    df1 = pd.read_table(trainF)
    df2 = pd.read_table(testF)
    df3 = pd.read_table(sortF)
    user_train_df = df1.drop(['tweet_id', 'time', 'content', 'hashtag'], axis=1)
    user_set = set(list(user_train_df['user_id']))
    user_test_df = df2.drop(['tweet_id', 'time', 'hashtag'], axis=1)

    success = 0
    for user in user_set:                         # 重复出现的user要记得筛掉
        # tag_list = random_rec(df3)
        tag_list = popular_rec(df3)
        # tag_list = latest_rec(user, df1)
        for tag in tag_list:
            if tag in Preproce.get_hashtag(user_test_df[user_test_df['user_id'] == user]['content']):
                success += 1
                break

    # print("random recommendation: "+str(success/len(user_set)))
    print("popularity recommendation: " + str(success / len(user_set)))
    # print("latest recommendation: " + str(success / len(user_set)))


if __name__ == '__main__':
    eval_rec(trainEle4, testEle5, sortedTag)

