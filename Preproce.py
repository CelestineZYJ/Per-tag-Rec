import pandas as pd
import re
import matplotlib.pyplot as plt
path1 = "./data/tweetTag.txt"
path2 = "./data/plusTag.txt"
df1 = pd.read_table(path1, names=['tweet_id', 'user_id', 'time', 'content'])
df2 = pd.read_table(path2)


def get_hashtag(content):
    """
    Get the hashtag from the content.
    """
    words = content.split()
    hashtag = [word for word in words if re.search(r'^#', word)]
    return hashtag


def get_tag_file(dataframe):
    dataframe['hashtag'] = dataframe['content'].apply(get_hashtag)
    dataframe.to_csv("./data/plusTag.txt", sep='\t', index=False)

    dataframe1 = dataframe.explode('hashtag')
    dataframe1.to_csv("./data/explodeTag.txt", sep='\t', index=False)

    dataframe2 = dataframe1.groupby(['hashtag'], as_index=False)['hashtag'].agg({'cnt':'count'})
    dataframe2 = dataframe2.sort_values(by=['cnt'], ascending=False)
    dataframe2.to_csv("./data/countTag.txt", sep='\t', index=False)


"""
def get_tag_hist(dataframe):
    list1 = dataframe['cnt'].values.tolist()
    print(list1)
    plt.hist(list1, 10)
    plt.show()
"""


def filter_single_user(dataframe):
    dataframe1 = dataframe.loc[0:876483]
    dataframe2 = dataframe.loc[876484:]

    # filter users in training set and test set simultaneously
    dataframe1 = dataframe1[dataframe1['user_id'].isin(dataframe2['user_id'].tolist())]
    dataframe2 = dataframe2[dataframe2['user_id'].isin(dataframe1['user_id'].tolist())]

    dataframe1.to_csv("./data/trainSet.txt", sep='\t', index=False)
    dataframe2.to_csv("./data/testSet.txt", sep='\t', index=False)


if __name__ == "__main__":
    # get_tag_file(df1)
    filter_single_user(df2)