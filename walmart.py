import streamlit as st
import pandas as pd
import numpy as np 

import pickle

st.title('WallMart Dataset')
df=pd.read_csv("file.csv")
st.write(df.head())
day = st.selectbox(
    'Select the Day',
    ('Monday', 'Tuesday', 'Wednesday','Thursday','Friday','Saturday','Sunday'))

st.write('You selected:', day)

department = st.selectbox(
    'Select the Department',
    ('PRODUCE', 'CANDY, TOBACCO, COOKIES', 'GROCERY DRY GOODS',
       'PERSONAL CARE', 'HOUSEHOLD CHEMICALS/SUPP', 'MENS WEAR',
       'PHARMACY OTC', 'CELEBRATION', 'DSD GROCERY', '1-HR PHOTO',
       'BEDDING', 'FINANCIAL SERVICES', 'BEAUTY', 'HOUSEHOLD PAPER GOODS',
       'PAINT AND ACCESSORIES', 'DAIRY', 'JEWELRY AND SUNGLASSES',
       'LADIESWEAR', 'COMM BREAD', 'FROZEN FOODS',
       'MEAT - FRESH & FROZEN', 'INFANT CONSUMABLE HARDLINES',
       'HOME DECOR', 'SERVICE DELI', 'SPORTING GOODS', 'COOK AND DINE',
       'IMPULSE MERCHANDISE', 'INFANT APPAREL', 'LIQUOR,WINE,BEER',
       'SEAFOOD', 'BRAS & SHAPEWEAR', 'HOME MANAGEMENT', 'AUTOMOTIVE',
       'WIRELESS', 'BOYS WEAR', 'PETS AND SUPPLIES', 'OPTICAL - FRAMES',
       'BATH AND SHOWER', 'SHOES', 'HARDWARE',
       'GIRLS WEAR, 4-6X  AND 7-14', 'ELECTRONICS', 'BAKERY',
       'OFFICE SUPPLIES', 'TOYS', 'LAWN AND GARDEN', 'PRE PACKED DELI',
       'ACCESSORIES', 'FABRICS AND CRAFTS', 'SWIMWEAR/OUTERWEAR',
       'MEDIA AND GAMING', 'HORTICULTURE AND ACCESS',
       'BOOKS AND MAGAZINES', 'PLUS AND MATERNITY',
       'PLAYERS AND ELECTRONICS', 'SLEEPWEAR/FOUNDATIONS', 'LADIES SOCKS',
       'LARGE HOUSEHOLD GOODS', 'FURNITURE', 'CAMERAS AND SUPPLIES',
       'OPTICAL - LENSES', 'MENSWEAR', 'SHEER HOSIERY', 'PHARMACY RX',
       'SEASONAL', 'OTHER DEPARTMENTS', 'CONCEPT STORES',
       'HEALTH AND BEAUTY AIDS'))

st.write('You selected : ', department)

x=df[["Weekday","DepartmentDescription"]]
y=df.iloc[:,0]


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
x["DepartmentDescription_label"]=encoder.fit_transform(x["DepartmentDescription"])
x["Weekday_label"]=encoder.fit_transform(x["Weekday"])

dictionary1=dict(zip((x["DepartmentDescription"].unique()),(x["DepartmentDescription_label"].unique())))
dictionary2=dict(zip((x["Weekday"].unique()),(x["Weekday_label"].unique())))

x1=x[["Weekday_label","DepartmentDescription_label"]]

seed=7
testsize=0.33
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=testsize,random_state=seed)


loaded_model = pickle.load(open('dtc_1.pkl','rb'))

t1=str(day)
t2=str(department)
t1=dictionary2[t1]
t2=dictionary1[t2]
t1,t2

features = np.array([[t1, t2]])

st.write("predicted: ",loaded_model.predict(features))
