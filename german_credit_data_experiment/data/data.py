import pandas as pd
credit= pd.read_csv("data/german_credit_data_raw.csv",header = 0, names = ['Index', 'Age', 'Sex', 'Job', 'Housing', 'Saving accounts','Checking account', 'Credit amount', 'Duration', 'Purpose', 'Default'])
credit=pd.DataFrame(data=credit, columns=["Index","Sex", "Age","Credit amount","Duration","Default"])
credit.to_csv('data/german_credit_data.csv',index=False)


if __name__=="__main__":
    print(credit.head())
    print(credit.info())