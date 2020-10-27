import pandas as pd
import os
import json

trainWeibo = './data/trainWeibo.txt'
testWeibo = './data/testWeibo.txt'


def list_to_json(lis, filename, filepath):
    os.chdir(filepath)
    with open(filename, 'w', encoding="utf-8") as f:
        for row in lis:
            js = json.dumps(row, ensure_ascii=False)
            f.write(js)
            f.write('\n')


def filter_hashtag():
    weibo = []
    file = open('./data/weibo_data2.txt', 'r', encoding='utf-8')
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


def filter_both_user(df1, df2, label, file1, file2):
    df_train = df1[df1['user_id'].isin(df2['user_id'].tolist())]
    df_test = df2[df2['user_id'].isin(df1['user_id'].tolist())]
    df_train.to_csv(file1, sep='\t', index=False)
    df_test.to_csv(file2, sep='\t', index=False)

    # data analysis
    print('number of train user: '+str(df_train.groupby(['user_id'], as_index=False)['user_id'].agg({'cnt': 'count'}).shape[0]))
    print('number of test user: '+str(df_test.groupby(['user_id'], as_index=False)['user_id'].agg({'cnt': 'count'}).shape[0]))
    print('number of train weibo: '+str(df_train.shape[0]))
    print('number of test weibo: '+str(df_test.shape[0]))
    train_tag = df_train.explode(label).groupby([label], as_index=False)[label].agg({'cnt': 'count'})
    test_tag = df_test.explode(label).groupby([label], as_index=False)[label].agg({'cnt': 'count'})
    print('number of train hashtags: '+str(train_tag.shape[0]))
    print('number of test hashtags: ' + str(test_tag.shape[0]))
    print('number of overlap hashtags: '+str(train_tag[train_tag[label].isin(test_tag[label].tolist())].shape[0]))


def divide_train_test(df, file1, file2):
    label = 'topics'
    df_train = df[1:2034202]
    df_test = df[2034203:]
    df_train = filter_5_weibo(df_train)
    filter_both_user(df_train, df_test, label, file1, file2)


if __name__ == "__main__":
    divide_train_test(filter_hashtag(), trainWeibo, testWeibo)

