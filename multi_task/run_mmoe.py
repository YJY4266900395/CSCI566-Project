# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *
import numpy as np
from sklearn.metrics import mean_squared_error, log_loss, roc_auc_score
import sys

if __name__ == "__main__":

    # data description can be found in https://www.biendata.xyz/competition/icmechallenge2019/
    data = pd.read_csv(
        "../data/interaction_part.csv",
        sep=",",
        # names=[
        # "user_id", "pid", "author_id", "category_id", "parent_id",
        # "root_id", "exposed_time", "author_fan",
        # "watch_time", "duration", "cvm_like", "effective_view",
        # "comment", "follow", "collect", "forward",
        # "hate", "tag_name", "title",
        # "p_hour", "p_date", "gender", "age",
        # "mod_price", "city", "community_type", "city_level"
        # ]
    )

    data = data.rename(columns={"click": "effective_view"})

    sparse_features = [
        "user_id", "pid", "author_id", "category_id",
        "parent_id", "root_id", "gender",
        "fre_city", "fre_community_type", "fre_city_level",
        "tag_name", "title"
    ]

    dense_features = [
        "exposed_time", "author_fans_count", "duration",
        "p_hour", "p_date", "age", "mod_price"
    ]

    target = [
        'watch_time', 'effective_view', 'cvm_like',
        'comment', 'follow', 'collect', 'forward', 'hate'
    ]

    for t in target:
        if t != "watch_time":
            data[t] = data[t].astype(int)

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name
    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=4)
        for feat in sparse_features
    ] + [
        DenseFeat(feat, 1)
        for feat in dense_features
    ]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns
    )

    # 3.generate input data for model
    split_boundary = int(data.shape[0] * 0.8)
    train, test = data[:split_boundary], data[split_boundary:]

    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = MMOE(
        dnn_feature_columns,
        num_experts=8,  # expert数
        expert_dnn_hidden_units=(64, 32),  # expert
        gate_dnn_hidden_units=(64, ),  # gate
        tower_dnn_hidden_units=(64, 32),  # tower
        task_types=['regression'] + ['binary'] * 7,
        task_names=tuple(target),
        l2_reg_embedding=1e-5,
        l2_reg_dnn=0,
        dnn_dropout=0.1,
        dnn_activation='relu',
        dnn_use_bn=False,
        seed=1024,
        device=device
    )

    model.compile(
        "adagrad",
        loss=['mse'] + ['binary_crossentropy'] * 7,
        metrics=None,
    )

    history = model.fit(
        train_model_input,
        train[target].values,
        batch_size=32,
        epochs=10,
        verbose=2
    )

    ### test
    pred = model.predict(test_model_input, 256)
    if isinstance(pred, list):
        pred = np.concatenate([p.reshape(-1, 1) for p in pred], axis=1)

    for i, target_name in enumerate(target):
        y_true = test[target_name].values
        y_pred = pred[:, i]

        if target_name == "watch_time":
            print(
                f"{target_name} test MSE",
                round(mean_squared_error(y_true, y_pred), 4)
            )
            continue

        if len(np.unique(y_true)) < 2:
            print(f"{target_name} test LogLoss N/A (only one class)")
            print(f"{target_name} test AUC N/A (only one class)")
            continue

        print(
            f"{target_name} test LogLoss",
            round(log_loss(y_true, y_pred), 4)
        )
        print(
            f"{target_name} test AUC",
            round(roc_auc_score(y_true, y_pred), 4)
        )
