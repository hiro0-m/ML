import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib


# one-hotエンコーディング
def one_hot_encoding(X, obj_ary):
    one_df = pd.get_dummies(X, dummy_na=True, columns=obj_ary)
    return one_df

# データの整合
def check_columns(X_model, X_score):
    
    model_cols = pd.DataFrame(None, columns=X_model.columns.values)
    X_score_after = pd.concat([model_cols,X_score])

    X_score_after.loc[:,list(set(X_model.columns.values) - set(X_score.columns.values))] = \
        X_score_after.loc[:,list(set(X_model.columns.values) - set(X_score.columns.values))].fillna(0,axis=0)

    X_score_after = X_score_after.drop(list(set(X_score.columns.values) - set(X_model.columns.values)),axis=1)
    
    X_score_after = X_score_after.reindex(X_model.columns.values, axis=1)

    return X_score_after

# モデルの保存
def save_model(model,name):
    if os.path.isdir('./model') == False:
        os.mkdir('./model')
    joblib.dump(model,'./model/' + name + '.pkl')

# 学習と評価
def learn_and_score(X_train, y_train, scoring='roc_auc',filename=''):
    
    pipeline = {
    'logistic':
        Pipeline([('scl',StandardScaler()),('est',LogisticRegression(random_state=1))]),
    'tree':
        Pipeline([('est',DecisionTreeClassifier(random_state=1))]),
    'rf':
        Pipeline([('est',RandomForestClassifier(random_state=1))]),
    'gb':
        Pipeline([('est',GradientBoostingClassifier(random_state=1))])
    }
    
    scores = {}
    for pipe_name, pipeline in pipeline.items():
        pipeline.fit(X_train, y_train)
        #モデルを保存
        save_model(pipeline,filename + '_' +pipe_name + '_learned')
        scores[pipe_name] = np.mean(cross_val_score(pipeline, X_train, y_train, scoring=scoring, cv=5))
        

    return scores

#モデルのロード
def load_model(modelname):
    load_model = joblib.load(modelname)
    return load_model

def main():
    #ファイル名は修正して使用する。
    print('学習データは [./data]配下に格納してください')
    print('学習データのファイル名(csv)を入力してください')
    train_file = input('>> ')
    print('検証データは [./data]配下に格納してください')
    print('検証データのファイル名(csv)を入力してください')
    test_file = input('>> ')
    drop_columns=list()

    df = pd.read_csv('./data/' + train_file, ',')
    df_test = pd.read_csv('./data/' + test_file, ',')

    print(df.nunique(dropna=False))

    print('正解データの列名を入力してください')
    print(df.columns.values)
    ans_col = list()
    ans_col_name = input('>> ')
    ans_col.append(ans_col_name)
    
    #正解データ列の指定
    y_train = df.loc[:,ans_col]

    print('検証データのid名を入力してください')
    print(df_test.columns.values)
    test_id = input('>> ')
    y_id = pd.DataFrame(df_test.loc[:,test_id])
    
    print(df.dtypes)
    print('不要な列名を入力してください(複数ある場合は「,」で区切って入力)')
    print(df.columns.values)
    drop_col = input('>> ')

    if drop_col != '':
        drop_index = drop_col.find(',')
        if drop_index != -1:
            drop_columns = drop_col.split(',')
        else:
            drop_columns = drop_col
    
    #不要列削除
    if drop_columns != '':
        df = df.drop(drop_columns,axis=1)
        df_test = df_test.drop(drop_columns,axis=1)

    #学習データから正解データを削除
    df = df.drop(ans_col,axis=1)

    #カテゴリ変数
    list_category = list()
    for category_columns in df.columns:
        if df[category_columns].dtypes == object:
            list_category.append(category_columns)

    print('カテゴリ変数')
    print(list_category)

    #####################
    #----- モデル用 -----#
    ####################
    print('モデル用の前処理開始')
    df_ohe = one_hot_encoding(df,list_category)
    print('ワンホットエンコーディング後サイズ：' + str(df_ohe.shape))
    imp = SimpleImputer()
    imp.fit(df_ohe)
    df_ohe = pd.DataFrame(imp.transform(df_ohe), columns=df_ohe.columns.values)

    rf = RandomForestClassifier(random_state=1)
    rf.fit(df_ohe, y_train)
    
    #特徴選択
    #select = RFECV(RandomForestClassifier(n_estimators=100, random_state=1), min_features_to_select=10,step=0.05)
    select = RFECV(estimator=rf)
    select.fit(df_ohe, y_train)

    #特徴選択後のサイズ
    X_train = select.transform(df_ohe)
    X_train = pd.DataFrame(X_train, columns=df_ohe.columns.values[select.support_])
    print('前処理完了後サイズ：' + str(X_train.shape))

    #重要度
    importances = pd.DataFrame({"features":df_ohe.columns,"importances":rf.feature_importances_,"select":select.support_})
    print(importances)


    
    #####################
    #----- スコア用 -----#
    ####################
    print('スコア用の前処理開始')
    df_test_ohe = one_hot_encoding(df_test,list_category)

    print('ワンホットエンコーディング後サイズ：' + str(df_test_ohe.shape))

    # モデルと整合を合わせる
    X_test = check_columns(X_train,df_test_ohe)

    imp.fit(X_test)
    X_test = pd.DataFrame(imp.transform(X_test), columns=X_test.columns.values)

    print('前処理完了後サイズ：' + str(X_test.shape))

    #select_score = 'f1'
    select_score = 'roc_auc'

    scores = learn_and_score(X_train, y_train, select_score, train_file[:-4])
    
    print('選択評価指標：' + select_score)
    print('######## 評価結果 ########')
    print(pd.Series(scores).sort_values(ascending=False))

    print('検証に使用するモデルの略称入力してください')
    
    for key in scores.keys():
        print(key)
        
    
    modelname = ''
    model_num = input('>>')
    if model_num != '':
        modelname = model_num
    
    model = load_model('./model/'+ train_file[:-4] + '_' + modelname + '_learned.pkl')
    #予想確率
    #pre = pd.DataFrame(model.predict_proba(X_test), columns=ans_col)
    #予測
    pre = pd.DataFrame(model.predict(X_test), columns=ans_col)
    score = y_id.join(pre)
    if os.path.isdir('./pred') == False:
        os.mkdir('./pred')
    
    pred_name = './pred/' + test_file[:-4] + '_' + modelname +'_pred.csv'
    score.to_csv(pred_name,index=False)

    print('検証結果を' + pred_name + 'に保存しました')

if __name__ == '__main__':
    main()