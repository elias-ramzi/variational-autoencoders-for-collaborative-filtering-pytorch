import os

import numpy as np
import pandas as pd
from scipy import sparse
import pickle


# DATASET = 'ml-latest-small-trans'
# with open(os.path.join('/share/DEEPLEARNING/datasets/graph_datasets/movielens/', 'train.txt')) as f:
#     train = f.read()
# train = train.split('\n')
# train.remove('')
# with open(os.path.join('/share/DEEPLEARNING/datasets/graph_datasets/movielens/', 'test.txt')) as f:
#     test = f.read()
# test = test.split('\n')
# test.remove('')
#
# train_uid = []
# train_mid = []
# for line in map(lambda x: x.split(' '), train):
#     uid = int(line[0])
#     train_uid.extend([uid] * len(line[1:]))
#     train_mid.extend(map(int, line[1:]))
#
# test_uid = []
# test_mid = []
# for line in map(lambda x: x.split(' '), test):
#     uid = int(line[0])
#     test_uid.extend([uid] * len(line[1:]))
#     test_mid.extend(map(int, line[1:]))
#
# train = pd.DataFrame(list(zip(train_uid, train_mid)), columns=['uid', 'sid'])
# test = pd.DataFrame(list(zip(test_uid, test_mid)), columns=['uid', 'sid'])


# DATASET = 'gowalla_trans'
# train = pd.read_csv(os.path.join('/share/DEEPLEARNING/datasets/graph_datasets/gowalla_vae/trans', 'train.csv'))
# test = pd.read_csv(os.path.join('/share/DEEPLEARNING/datasets/graph_datasets/gowalla_vae/trans', 'test.csv'))

# DATASET = 'yelp2018_trans'
# train = pd.read_csv(os.path.join('/share/DEEPLEARNING/datasets/graph_datasets/yelp2018_vae/trans', 'train.csv'))
# test = pd.read_csv(os.path.join('/share/DEEPLEARNING/datasets/graph_datasets/yelp2018_vae/trans', 'test.csv'))

# DATASET = 'amazon_book_trans'
# REMOVE_USERS = True
# REMOVE_TEST_ITEMS = True
# train = pd.read_csv(os.path.join('splits/amazon-book/files_split/transductive', 'train.csv'))
# test = pd.read_csv(os.path.join('splits/amazon-book/files_split/transductive', 'test.csv'))

DATASET = 'ml_1m_trans'
REMOVE_USERS = True
REMOVE_TEST_ITEMS = True
train = pd.read_csv(os.path.join('splits/ml_1m/files_split/transductive', 'train.csv'))
test = pd.read_csv(os.path.join('splits/ml_1m/files_split/transductive', 'test.csv'))

os.makedirs(DATASET, exist_ok=True)


val_struct = train.copy()
test_struct = train.copy()
val_pred = test.copy()
test_pred = test.copy()

if REMOVE_USERS:
    users_to_remove = set(train['uid'].unique().tolist()) - set(test['uid'].unique().tolist())
    train = train[~train['uid'].isin(users_to_remove)]
    val_struct = val_struct[~val_struct['uid'].isin(users_to_remove)]
    test_struct = test_struct[~test_struct['uid'].isin(users_to_remove)]

if REMOVE_TEST_ITEMS:
    items_to_remove = set(test['sid'].unique().tolist()) - set(train['sid'].unique().tolist())
    test = test[~test['sid'].isin(items_to_remove)]
    val_pred = val_pred[~val_pred['sid'].isin(items_to_remove)]
    test_pred = test_pred[~test_pred['sid'].isin(items_to_remove)]

users = sorted(train['uid'].unique().tolist())
items = sorted(train['sid'].unique().tolist())
assert set(users) == set(test_pred['uid'].unique().tolist())
# users_val = sorted(val_struct['uid'].unique().tolist())
# users_test = sorted(test_struct['uid'].unique().tolist())
# n_users = len(users_train) + len(users_val) + len(users_test)
new_users = {x: new_x for new_x, x in enumerate(users)}
new_items = {x: new_x for new_x, x in enumerate(items)}
# new_users_val = {x: new_x for new_x, x in enumerate(users_val, start=len(users_train))}
# new_users_test = {x: new_x for new_x, x in enumerate(users_test, start=len(users_train)+len(users_val))}
# assert len(new_users_train) == len(users_train)
# assert len(new_users_train) == len(users_val)
# assert len(new_users_train) == len(users_test)
train['uid'] = train['uid'].map(new_users)
val_struct['uid'] = val_struct['uid'].map(new_users)
val_pred['uid'] = val_pred['uid'].map(new_users)
test_struct['uid'] = test_struct['uid'].map(new_users)
test_pred['uid'] = test_pred['uid'].map(new_users)

train['sid'] = train['sid'].map(new_items)
val_struct['sid'] = val_struct['sid'].map(new_items)
val_pred['sid'] = val_pred['sid'].map(new_items)
test_struct['sid'] = test_struct['sid'].map(new_items)
test_pred['sid'] = test_pred['sid'].map(new_items)


train.to_csv(os.path.join('splits/', 'train_reindex.csv'))
val_struct.to_csv(os.path.join('splits/', 'validation_tr_reindex.csv'))
val_pred.to_csv(os.path.join('splits/', 'validation_te_reindex.csv'))
test_struct.to_csv(os.path.join('splits/', 'test_tr_reindex.csv'))
test_pred.to_csv(os.path.join('splits/', 'test_te_reindex.csv'))


def save_weights_pkl(fname, weights):
    with open(fname, 'wb') as f:
        pickle.dump(weights, f, pickle.HIGHEST_PROTOCOL)


def load_train_data(csv_file):
    tp = pd.read_csv(csv_file)
    n_users = tp['uid'].max() + 1
    n_items = len(pd.unique(tp['sid']))

    rows, cols = tp['uid'], tp['sid']
    data = sparse.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float64', shape=(n_users, n_items))
    return data, n_items


train_data_csr, n_items = load_train_data(os.path.join('splits/', 'train_reindex.csv'))


def load_tr_te_data(csv_file_tr, csv_file_te, n_items):
    tp_tr = pd.read_csv(csv_file_tr)
    tp_te = pd.read_csv(csv_file_te)

    start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
    end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())
    assert pd.unique(tp_tr["uid"]).shape[0] == end_idx - start_idx + 1
    assert pd.unique(tp_te["uid"]).shape[0] == end_idx - start_idx + 1

    rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
    rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

    data_tr = sparse.csr_matrix((np.ones_like(rows_tr), (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    data_te = sparse.csr_matrix((np.ones_like(rows_te), (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, n_items))
    return data_tr, data_te


vad_data_tr_csr, vad_data_te_csr = load_tr_te_data(
    os.path.join('splits/', 'validation_tr_reindex.csv'),
    os.path.join('splits/', 'validation_te_reindex.csv'),
    n_items,
)

test_data_tr_csr, test_data_te_csr = load_tr_te_data(
    os.path.join('splits/', 'test_tr_reindex.csv'),
    os.path.join('splits/', 'test_te_reindex.csv'),
    n_items,
)

fname = os.path.join(DATASET, 'data_csr.pkl')
datas = [train_data_csr, vad_data_tr_csr, vad_data_te_csr, test_data_tr_csr, test_data_te_csr]
save_weights_pkl(fname, datas)
