import pandas as pd
import re

path1 = "./data/tweetTag.txt"
path2 = "./data/plusTag.txt"
df1 = pd.read_table(path1, names=['tweet_id', 'user_id', 'time', 'content'])
df2 = pd.read_table(path2)


def get_hashtag(content):
    """
    Get the hashtag from the content.
    """
    words = re.split(r'[:,.! "\']', str(content))
    hashtag = [word for word in words if re.search(r'^#', word)]
    # print(type(hashtag[0]))
    return hashtag


def get_tag_file(dataframe):
    dataframe['hashtag'] = dataframe['content'].apply(get_hashtag)
    dataframe.to_csv("./data/plusTag.txt", sep='\t', index=False)

    dataframe1 = dataframe.explode('hashtag')
    dataframe1.to_csv("./data/explodeTag.txt", sep='\t', index=False)

    dataframe2 = dataframe1.groupby(['hashtag'], as_index=False)['hashtag'].agg({'cnt': 'count'})
    dataframe2 = dataframe2.sort_values(by=['cnt'], ascending=False)
    dataframe2.to_csv("./data/countTag.txt", sep='\t', index=False)


def get_train_content(f):
    df = pd.read_table(f)
    df = df.drop(['tweet_id', 'user_id', 'time', 'hashtag'], axis=1)
    print(df)
    df.to_csv('./data/trainContent.txt')


def filter_single_user(dataframe):
    dataframe1 = dataframe.loc[0:876483]
    dataframe2 = dataframe.loc[876484:]

    # filter users in training set and test set simultaneously
    dataframe1 = dataframe1[dataframe1['user_id'].isin(dataframe2['user_id'].tolist())]
    dataframe2 = dataframe2[dataframe2['user_id'].isin(dataframe1['user_id'].tolist())]

    # dataframe1.to_csv("./data/trainSet.txt", sep='\t', index=False)
    # dataframe2.to_csv("./data/testSet.txt", sep='\t', index=False)

    # filter users whose tweets are more than 5 in training set
    dataframe3 = dataframe1.groupby(['user_id'], as_index=False)['user_id'].agg({'cnt': 'count'})
    dataframe3 = dataframe3.loc[dataframe3['cnt'] >= 5]

    dataframe4 = dataframe1[dataframe1['user_id'].isin(dataframe3['user_id'].tolist())]
    dataframe5 = dataframe2[dataframe2['user_id'].isin(dataframe3['user_id'].tolist())]

    dataframe4.to_csv("./data/trainSet.txt", sep='\t', index=False)
    dataframe5.to_csv("./data/testSet.txt", sep='\t', index=False)

    # get the hashtags sorted by count appearing in train set
    dataframe4 = dataframe4.drop(['hashtag'], axis=1)
    dataframe4['hashtag'] = dataframe4['content'].apply(get_hashtag)
    dataframe4 = dataframe4.explode('hashtag').groupby(['hashtag'], as_index=False)['hashtag'].agg({'cnt': 'count'})
    dataframe6 = dataframe4.sort_values(by=['cnt'], ascending=False)
    dataframe6.to_csv("./data/countTrainTag.txt", sep='\t', index=False)

    # get the hashtags sorted by count appearing in test set
    dataframe5 = dataframe5.drop(['hashtag'], axis=1)
    dataframe5['hashtag'] = dataframe5['content'].apply(get_hashtag)
    dataframe5 = dataframe5.explode('hashtag').groupby(['hashtag'], as_index=False)['hashtag'].agg({'cnt': 'count'})
    dataframe7 = dataframe5.sort_values(by=['cnt'], ascending=False)
    dataframe7.to_csv("./data/countTestTag.txt", sep='\t', index=False)

    # calculate the overlap number of hashtag both in train and test set
    train_tag_list = dataframe6['hashtag'].tolist()
    test_tag_list = dataframe7['hashtag'].tolist()
    print(len(set(train_tag_list) & set(test_tag_list)))


if __name__ == "__main__":
    # get_tag_file(df1)
    filter_single_user(df2)
    # get_train_content('./data/trainSet.txt')
    # get_hashtag('#lalala i want to #yes go')
