__author__ = 'foursking'
from gen_feat import make_train_set
from gen_feat import make_test_set
from sklearn.model_selection import train_test_split
import xgboost as xgb
from gen_feat import report
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd


def xgboost_with_multiply_month():
    seed = 22
    sub_start_date = '2016-03-15'
    sub_end_date = '2016-04-16'
    train_start_dates = ['2016-02-14', '2016-02-19', '2016-02-24', '2016-02-29', '2016-03-05', '2016-03-10']
    train_set = None
    labels = None
    for train_start_date in train_start_dates:
        train_end_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=32)
        train_end_date = train_end_date.strftime('%Y-%m-%d')
        test_start_date = train_end_date
        test_end_date = datetime.strptime(train_start_date, '%Y-%m-%d') + timedelta(days=37)
        test_end_date = test_end_date.strftime('%Y-%m-%d')
        user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date, sample=0.1, seed=22)
        #training_data = training_data.values
        #label = label.values #
        if train_set is None:
            train_set = training_data
            labels = label
        else:
            train_set = np.concatenate((train_set, training_data), axis=0)
            labels = np.concatenate((label, labels), axis=0)
    print train_set.shape
    X_train, X_test, y_train, y_test = train_test_split(train_set, labels, test_size=0.2, random_state=0)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 3,

            'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
            'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 111
    param['nthread'] = 4
    plst = param.items()
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst=xgb.train(plst, dtrain, num_round, evallist)
    sub_user_index, sub_trainning_data = make_test_set(sub_start_date, sub_end_date)
    sub_trainning_data = xgb.DMatrix(sub_trainning_data.values)
    y = bst.predict(sub_trainning_data)
    sub_user_index['label'] = y
    pred = sub_user_index
    pred['user_id'] = pred['user_id'].astype(int)
    pred.to_csv('./sub/submission.csv', index=False, index_label=False)

        #print train_start_date, train_end_date, test_begin_date, test_end_date

def xgboost_make_submission_v1():
    train_start_date = '2016-03-10'
    train_end_date = '2016-04-11'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-03-15'
    sub_end_date = '2016-04-16'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(training_data.values, label.values, test_size=0.2, random_state=0)
    dtrain=xgb.DMatrix(X_train, label=y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    param = {'learning_rate' : 0.1, 'n_estimators': 1000, 'max_depth': 3, 
        'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
        'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 283
    param['nthread'] = 4
    #param['eval_metric'] = "auc"
    plst = param.items()
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst=xgb.train(plst, dtrain, num_round, evallist)
    sub_user_index, sub_trainning_data = make_test_set(sub_start_date, sub_end_date)
    sub_trainning_data = xgb.DMatrix(sub_trainning_data.values)
    y = bst.predict(sub_trainning_data)
    sub_user_index['label'] = y
    pred = sub_user_index[sub_user_index['label'] >= 0.03]
    pred = pred[['user_id', 'sku_id']]
    pred = pred.groupby('user_id').first().reset_index()
    pred['user_id'] = pred['user_id'].astype(int)
    pred.to_csv('./sub/submission.csv', index=False, index_label=False)



def xgboost_cv():
    train_start_date = '2016-03-05'
    train_end_date = '2016-04-06'
    test_start_date = '2016-04-11'
    test_end_date = '2016-04-16'

    sub_start_date = '2016-02-05'
    sub_end_date = '2016-03-05'
    sub_test_start_date = '2016-03-05'
    sub_test_end_date = '2016-03-10'

    user_index, training_data, label = make_train_set(train_start_date, train_end_date, test_start_date, test_end_date)
    X_train, X_test, y_train, y_test = train_test_split(training_data, label, test_size=0.2, random_state=0)
    dtrain=xgb.DMatrix(X_train, label=y_train)
    dtest=xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 10, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 4000
    param['nthread'] = 4
    param['eval_metric'] = "auc"
    plst = param.items()
    plst += [('eval_metric', 'logloss')]
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    bst=xgb.train( plst, dtrain, num_round, evallist)

    sub_user_index, sub_trainning_date, sub_label = make_train_set(sub_start_date, sub_end_date,
                                                                   sub_test_start_date, sub_test_end_date)
    test = xgb.DMatrix(sub_trainning_date)
    #y = bst.predict(test)

    pred = sub_user_index.copy()
    y_true = sub_user_index.copy()
    pred['label'] = y
    y_true['label'] = label
    report(pred, y_true)


if __name__ == '__main__':
    #xgboost_cv()
    #xgboost_make_submission()
    xgboost_with_multiply_month()
