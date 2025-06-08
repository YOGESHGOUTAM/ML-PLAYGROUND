import streamlit as st
import pandas as pd
from sklearn.mode_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

st.title("ML PLAYGROUND")

uploaded_file=st.file_uploader("Drop your File Here!!",type=["csv"])

if uploaded_file:
	df=pd.read_csv(uploaded_file)
	st.subheader("Dataset Preview")
	st.dataframe(df.head())
	
	target=st.selectbox("Select Target Column", df.columns)
	
	#NOW SELECT FEATURES
	 
	features_columns=[col for col in df.columns if col!=target]

	features=st.multiselect("Select Feature Columns",feature_columns,default=feature_columns)

    if st.button("Train Model"):
        X=df[features]
        y=df[target]

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

        model=RandomForestClassifier()
        model.fit(X_train,y_train)

        y_pred=model.predict(X_test)

        st.subheader("Classification Report")
        st.text(classification_report(y_test,y_pred))
        
        
        
        
