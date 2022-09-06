import os

import numpy as np
import pandas as pd
from scipy import sparse
import pickle

train = pd.read_csv(os.path.join('/share/DEEPLEARNING/datasets/graph_datasets/movielens/files_split/', 'train.csv'))
val_struct = pd.read_csv(os.path.join('/share/DEEPLEARNING/datasets/graph_datasets/movielens/files_split/', 'validation_tr.csv'))
val_pred = pd.read_csv(os.path.join('/share/DEEPLEARNING/datasets/graph_datasets/movielens/files_split/', 'validation_te.csv'))
test_struct = pd.read_csv(os.path.join('/share/DEEPLEARNING/datasets/graph_datasets/movielens/files_split/', 'test_tr.csv'))
test_pred = pd.read_csv(os.path.join('/share/DEEPLEARNING/datasets/graph_datasets/movielens/files_split/', 'test_te.csv'))

users_train = sorted(train['uid'].unique().tolist())
users_val = sorted(val_struct['uid'].unique().tolist())
users_test = sorted(test_struct['uid'].unique().tolist())
n_users = len(users_train) + len(users_val) + len(users_test)
new_users_train = {x: new_x for new_x, x in enumerate(users_train)}
new_users_val = {x: new_x for new_x, x in enumerate(users_val, start=len(users_train))}
new_users_test = {x: new_x for new_x, x in enumerate(users_test, start=len(users_train)+len(users_val))}
assert len(new_users_train) == len(users_train)
assert len(new_users_val) == len(users_val)
assert len(new_users_test) == len(users_test)
train['uid'] = train['uid'].map(new_users_train)
val_struct['uid'] = val_struct['uid'].map(new_users_val)
val_pred['uid'] = val_pred['uid'].map(new_users_val)
test_struct['uid'] = test_struct['uid'].map(new_users_test)
test_pred['uid'] = test_pred['uid'].map(new_users_test)

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
    print(n_items)

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

fname = os.path.join('ml-latest-small', 'data_csr.pkl')
datas = [train_data_csr, vad_data_tr_csr, vad_data_te_csr, test_data_tr_csr, test_data_te_csr]
save_weights_pkl(fname, datas)
