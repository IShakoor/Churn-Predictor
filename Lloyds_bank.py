import pandas as pd

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