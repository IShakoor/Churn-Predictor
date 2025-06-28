import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_recall_curve
import xgboost as xgb

# load data from Excel
customer_demographics = pd.read_excel('Customer_Churn_Data_Large.xlsx', sheet_name=0)
transaction_history = pd.read_excel('Customer_Churn_Data_Large.xlsx', sheet_name=1)
customer_service = pd.read_excel('Customer_Churn_Data_Large.xlsx', sheet_name=2)
online_activity = pd.read_excel('Customer_Churn_Data_Large.xlsx', sheet_name=3)
churn_status = pd.read_excel('Customer_Churn_Data_Large.xlsx', sheet_name=4)

# merge the data into a single table
def merge_customer_data(demographics = customer_demographics, transactions = transaction_history, service = customer_service, online = online_activity, churn = churn_status):

    # start with demographics as base
    df = demographics.copy()
    
    # merge all datasets
    df = df.merge(transactions, on='CustomerID', how='left')
    df = df.merge(service, on='CustomerID', how='left')
    df = df.merge(online, on='CustomerID', how='left')
    df = df.merge(churn, on='CustomerID', how='left')
    
    return df

# aggregating merged data
def aggregate_customer_level_data(df):
    agg_funcs = {
        'Age': 'first',
        'Gender': 'first',
        'MaritalStatus': 'first',
        'IncomeLevel': 'first',
        'TransactionID': lambda x: list(x.dropna().unique()),
        'TransactionDate': lambda x: list(pd.to_datetime(x.dropna())),
        'AmountSpent': [lambda x: list(x.dropna()), 'sum', 'count'],
        'ProductCategory': lambda x: list(x.dropna()),

        # interactions
        'InteractionID': [lambda x: list(x.dropna().unique()), 'count'],
        'InteractionDate': lambda x: list(pd.to_datetime(x.dropna())),
        'InteractionType': lambda x: list(x.dropna()),
        'ResolutionStatus': lambda x: list(x.dropna()),

        # online behaviour
        'LastLoginDate': 'max',
        'LoginFrequency': 'first',
        'ServiceUsage': 'first',

        # target variable
        'ChurnStatus': 'first'
    }

    customer_df = df.groupby('CustomerID').agg(agg_funcs)

    # flatten columns
    customer_df.columns = [
        '_'.join(col).strip('_') if isinstance(col, tuple) else col
        for col in customer_df.columns
    ]

    # rename to easier column names
    customer_df.rename(columns={
        'Age_first': 'Age',
        'Gender_first': 'Gender',
        'MaritalStatus_first': 'MaritalStatus',
        'IncomeLevel_first': 'IncomeLevel',
        'AmountSpent_<lambda_0>': 'AmountSpentList',
        'AmountSpent_sum': 'TotalSpent',
        'AmountSpent_count': 'NumTransactions',
        'TransactionID_<lambda>': 'TransactionIDList',
        'ProductCategory_<lambda>': 'ProductCategoryList',
        'InteractionID_<lambda>': 'InteractionIDList',
        'InteractionID_count': 'NumInteractions',
        'InteractionDate_<lambda>': 'InteractionDateList',
        'InteractionType_<lambda>': 'InteractionTypeList',
        'ResolutionStatus_<lambda>': 'ResolutionStatusList',
        'LastLoginDate_max': 'LastLoginDate',
        'LoginFrequency_first': 'LoginFrequency',
        'ServiceUsage_first': 'ServiceUsage',
        'ChurnStatus_first': 'ChurnStatus'
    }, inplace=True)

    customer_df = customer_df.reset_index()

    return customer_df

# flag high spenders
def flag_high_spenders(df, column='TotalSpent', percentile=0.99):
    df = df.copy()
    
    if column not in df.columns:
        print(f"Column '{column}' not found. Skipping high spender flag.")
        return df

    threshold = df[column].quantile(percentile)
    df['HighSpender'] = df[column] > threshold

    return df

# performing feature engineering
def add_engineered_features(df):
    df = df.copy()
    today = pd.to_datetime("2025-06-16") # fixed 

    # find the unresolved rate
    def calc_unresolved_rate(statuses):
        if not isinstance(statuses, (list, np.ndarray)):
            return 0
        if len(statuses) == 0:
            return 0
        unresolved = sum(1 for s in statuses if isinstance(s, str) and s.lower() == 'unresolved')
        return unresolved / len(statuses)

    df['UnresolvedInteractionRate'] = df['ResolutionStatusList'].apply(calc_unresolved_rate)

    # interaction recency (days)
    df['InteractionRecency'] = df['InteractionDateList'].apply(
        lambda dates: (today - max(dates)).days if dates else np.nan
    )

    # has interacted (binary)
    df['HasInteracted'] = df['NumInteractions'].apply(lambda x: int(x > 0))

    # avg transaction amount
    df['AvgSpentPerTransaction'] = df['TotalSpent'] / df['NumTransactions'].replace(0, np.nan)

    # transaction recency
    df['TransactionRecency'] = df['TransactionDate_<lambda>'].apply(
        lambda dates: (today - max(dates)).days if dates else np.nan
    )

    # unique categories
    df['UniqueProductCategories'] = df['ProductCategoryList'].apply(
        lambda lst: len(set(lst)) if isinstance(lst, list) else 0
    )

    # days since last login
    df['DaysSinceLastLogin'] = (today - pd.to_datetime(df['LastLoginDate'], errors='coerce')).dt.days

    return df

# remove unneccessary columns
def drop_unnecessary_columns(df):
    cols_to_drop = [
        'CustomerID',
        'TransactionIDList',
        'TransactionDate_<lambda>',
        'InteractionIDList',
        'InteractionID_<lambda_0>',
        'InteractionDateList',
        'InteractionTypeList',
        'ResolutionStatusList',
        'LastLoginDate',
        'ProductCategoryList',
        'AmountSpentList'
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    return df

# encoding categorical cols
def encode_categorical_features(df):
    df = df.copy()
    
    # categorical cols
    categorical_cols = [
        'Gender',
        'MaritalStatus',
        'IncomeLevel',
        'LoginFrequency',
        'ServiceUsage'
    ]

    # check cols exist
    cols_to_encode = [col for col in categorical_cols if col in df.columns]

    # encoding
    df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    return df_encoded

# normalise numerical features
def normalize_numerical_features(df):
    df = df.copy()
    
    # numerical cols
    numerical_cols = [
        'Age',
        'TotalSpent',
        'NumTransactions',
        'NumInteractions',
        'HighSpender',
        'UnresolvedInteractionRate',
        'InteractionRecency',
        'HasInteracted',
        'AvgSpentPerTransaction',
        'TransactionRecency',
        'UniqueProductCategories',
        'DaysSinceLastLogin'
    ]

    # check cols exist
    cols_to_scale = [col for col in numerical_cols if col in df.columns]

    # noralise cols
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    return df
