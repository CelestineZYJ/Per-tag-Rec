from pandas.io.json import json_normalize
import pandas as pd
import os
import json


def list_to_json(lis, filename, filepath):
    os.chdir(filepath)
    with open(filename, 'w', encoding="utf-8") as f:
        for row in lis:
            js = json.dumps(row, ensure_ascii=False)
            f.write(js)
            f.write('\n')


def filter_hashtag():
    weibo = []
    file = open('./data/weibo_data_month.json', 'r', encoding='utf-8')
    for line in file.readlines():
        dic = json.loads(line)
        if dic['topics']:
            weibo.append(dic)
    # list_to_json(tweets, 'weiboTag.json', './data')
    df = pd.DataFrame.from_dict(weibo, orient='columns')
    df.to_csv("./data/dfWeiboTag.txt", sep='\t', index=False)
    return df


def filter_5_weibo(df):
    # filter users whose tweets are more than 5 in training set
    df_user = df.groupby(['user_id'], as_index=False)['user_id'].agg({'cnt': 'count'})
    df_user = df_user.loc[df_user['cnt'] >= 5]
    df_5more = df[df['user_id'].isin(df_user['user_id'].tolist())]
    return df_5more


def filter_both_user(df1, df2):
    df_train = df1[df1['user_id'].isin(df2['user_id'].tolist())]
    df_test = df2[df2['user_id'].isin(df1['user_id'].tolist())]
    df_train.to_csv("./data/trainWeibo.txt", sep='\t', index=False)
    df_test.to_csv("./data/testWeibo.txt", sep='\t', index=False)
    print(df_train.groupby(['user_id'], as_index=False)['user_id'].agg({'cnt': 'count'}))
    print(df_test.groupby(['user_id'], as_index=False)['user_id'].agg({'cnt': 'count'}))
    #print(df_train)
    #print(df_test)


def divide_train_test(df):
    df_train = df[1:92066]
    df_test = df[92067:]
    # df_train = filter_5_weibo(df_train)
    # df_test = filter_5_weibo(df_test)
    filter_both_user(df_train, df_test)


if __name__ == "__main__":
    divide_train_test(filter_hashtag())

