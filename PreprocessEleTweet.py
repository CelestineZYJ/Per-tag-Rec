import pandas as pd
import PreproceWeibo
import Preproce
file_path1 = './data/election_tweets.txt'
file_path2 = './data/election_tag.txt'
file_path3 = "./data/dfEleTag.txt"
trainEle4 = './data/trainEle.txt'
testEle5 = './data/testEle.txt'
sortedTag = './data/sortTagEle.txt'


def sort_hashtag_list(f, f4):
    df = pd.read_table(f4)
    df['hashtag'] = df['content'].apply(Preproce.get_hashtag)
    df = df.explode('hashtag').groupby(['hashtag'], as_index=False)['hashtag'].agg({'cnt': 'count'})
    df = df.sort_values(by=['cnt'], ascending=False)
    print(df)
    df.to_csv(f, sep='\t', index=False)


# filter the tweets contains hashtag
def filter_tag_tweet(file1, file2):
    f1 = open(file1, 'r', encoding='utf-8')
    f2 = open(file2, 'w', encoding='utf-8')
    for line in f1:
        if '#' in str(line):
            f2.write(str(line))


# read file and write it to dataframe file
def str_to_file(file2, file3):
    df = pd.read_table(file2, sep='\|\|', names=['unknown1', 'tweet_id', 'repeat', 'unknown2', 'user_id', 'time', 'content', 'attitude'])
    df['hashtag'] = df['content'].apply(Preproce.get_hashtag)
    df.to_csv(file3, sep='\t', index=False)
    print(df)


# divide the train and test sets and filter 5 tweets and common user
def divide_train_test(file3, f4, f5):
    label = 'hashtag'
    df = pd.read_table(file3)
    df_train = df[1:276472]
    df_test = df[276473:]
    df_train = PreproceWeibo.filter_5_weibo(df_train)
    print(df_train)
    PreproceWeibo.filter_both_user(df_train, df_test, label, f4, f5)


if __name__ == '__main__':
    # str_to_file(file_path2, file_path3)
    # divide_train_test(file_path3, trainEle4, testEle5)
    sort_hashtag_list(sortedTag, trainEle4)
