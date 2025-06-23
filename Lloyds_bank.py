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
