import pandas as pd
import numpy as np
import time as t
import Preproce
np.random.seed(int(t.time()))

train_path = "./data/trainSet.txt"
test_path = "./data/testSet.txt"
freq_tag_path = "./data/countTag.txt"

train_df = pd.read_table(train_path)
test_df = pd.read_table(test_path)
freq_tag_df = pd.read_table(freq_tag_path)


def random_rec(dataframe):
    tag_rec = []
    for i in range(5):
        tag_rec.append(dataframe['hashtag'].loc[np.random.randint(100)])
    return tag_rec


def popular_rec(dataframe):
    tag_rec = []
    for i in range(5):
        tag_rec.append(dataframe['hashtag'].loc[i])
        return tag_rec


def latest_rec(user, dataframe):
    tag_rec = []
    user_df = dataframe
    for i in range(1):
        max_tweet_id = max(user_df['tweet_id'].loc[user_df['user_id'] == user])
        print(max_tweet_id)
        print(user_df['hashtag'].loc[user_df['tweet_id'] == max_tweet_id])


def eval_rec(dataframe1, dataframe2, dataframe3):
    user_train_df = dataframe1.drop(['tweet_id', 'time', 'hashtag'], axis=1)
    user_test_df = dataframe2.drop(['tweet_id', 'time', 'hashtag'], axis=1)

    success = 0

    for user in user_train_df['user_id']:                         # 重复出现的user要记得筛掉
        tag_list = random_rec(dataframe3)
        for tag in tag_list:
            # print(Preproce.get_hashtag(user_test_df[user_test_df['user_id'] == user]['content']))
            if tag in Preproce.get_hashtag(user_test_df[user_test_df['user_id'] == user]['content']):
                success += 1

    print(success)


if __name__ == "__main__":
    eval_rec(train_df, test_df, freq_tag_df)