import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",100)

df = pd.read_csv("rfmUsingData.csv",
                 sep=";",
                 decimal=',',
                 parse_dates=["Date"])

print(f"row count {df.shape[0]} ve attribute count {df.shape[1]}")

df.head()

df.isnull().sum()

len(df["Customer_id"].unique())

df["Bought"].value_counts()

df["Amount"].describe()

plt.style("ggplot")
plt.hist(df['Date'],bins=10,)
plt.grid(alpha=0.75)
plt.xlabel("Date")
plt.ylabel("Transaction Caount")
plt.show()

print(df['Date'].min(), df['Date'].max())

sonTarih = dt.datetime(2012,5,4)
df['Day_Dif']=sonTarih - df['Date']
df['Day_Dif'].astype('timedelta64[D]')
df['Day_Dif']=df['Day_Dif'] / np.timedelta64(1, 'D')
df.head()

plt.hist(df['Day_Dif'])
plt.grid(alpha=0.75)
plt.xlabel("Day Difference")
plt.ylabel("Transaction Count")
plt.show()

df=df[df['Day_Dif'] >= 1000]
print(f"row count: {df.shape[0]}")

rfmTable = df.groupby('Customer_id').agg(
    {'Day_Dif': lambda x:x.min(), # Recency
     'Customer_id': lambda x: len(x), # Frequency
     'Amount': lambda x: x.sum()}) # Monetary Value

rfmTable.rename(columns=
                {'Day_Dif': 'recency',
                 'Customer_id': 'frequency',
                 'Amount': 'monetary_value'},
                inplace=True)
rfmTable.head()

quart = rfmTable.quantile(q=[0.25,0.50,0.75]).to_dict()
print(quart)
# {
#   'recency':
#       {
#       0.25: 2131.0,
#       0.5: 3032.0,
#       0.75: 4170.0
#       },
#  'frequency':
#      {
#      0.25: 4.0,
#      0.5: 5.0,
#      0.75: 6.0
#      },
#  'monetary_value':
#      {
#      0.25: 79.61,
#      0.5: 104.04,
#      0.75: 134.9
#      }
# }

def RClass(x, p, d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4

def FMClass(x, p, d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]:
        return 2
    else:
        return 1

rfmSeg = rfmTable
rfmSeg['R_Quartile'] = rfmSeg['recency'].apply(RClass, args=('recency', quart,))
rfmSeg['F_Quartile'] = rfmSeg['frequency'].apply(FMClass, args=('frequency', quart,))
rfmSeg['M_Quartile'] = rfmSeg['monetary_value'].apply(FMClass, args=('monetary_value', quart,))


rfmSeg['RFMScore'] = rfmSeg.R_Quartile.map(str) \
                            + rfmSeg.F_Quartile.map(str) \
                            + rfmSeg.M_Quartile.map(str)
rfmSeg.RFMScore.head()
# Customer
# 1      112
# 2      121
# 3      343
# 4      111
# 5      444

rfmSeg.groupby('RFMScore').agg('monetary_value').mean().sort_index()
