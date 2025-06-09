import pandas as pd 
import streamlit as st
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df=pd.read_csv('C:\\datasets\\personality_dataset.csv')
df.dropna(inplace=True)

df['Social_event_attendance']=df['Social_event_attendance'].astype(int)
df['Time_spent_Alone']=df['Time_spent_Alone'].astype(int)
df['Going_outside']=df['Going_outside'].astype(int)
df['Friends_circle_size']=df['Friends_circle_size'].astype(int)
df['Post_frequency']=df['Post_frequency'].astype(int)
df['Stage_fear']=df['Stage_fear'].map({'Yes':1,'No':0})
df['Drained_after_socializing']=df['Drained_after_socializing'].map({'No':0,'Yes':1})
df['Personality']=df['Personality'].map({'Introvert':0,'Extrovert':1})

xtrain,xtest,ytrain,ytest=train_test_split(df.iloc[:,:-1],df.iloc[:,-1],test_size=0.2)

dtagain=DecisionTreeClassifier(criterion='gini',max_depth=2,min_samples_split=2)
dtagain.fit(xtrain,ytrain)
#dtpredagain=dtagain.predict(xtest)
#print(accuracy_score(ytest,dtpredagain))

st.title('QUICK  PERSONALITY  CHECK')

st.sidebar.title("I DON'T KNOW IF YOU'RE THE QUIET TYPE OR THE LIFE OF THE PARTY-BUT EITHER WAY, I'M ALREADY DRAWN TO YOUR ENERGY")
st.sidebar.title("TO KNOW MORE - ")
st.sidebar.header('GIVE ANSWERS TO THE MENTIONED QUESTIONS')


st.write(' ##### How much Time (in hours) you Spent Alone in a Day?')
time_spend=st.number_input("",value=0,key='input1') #max=12

st.write(' ##### Do you have Stage Fear?')
stage_fear=st.selectbox('',options=['YES','NO'],key='input2')
stage_fear_again=''
if stage_fear=='YES':
    stage_fear_again=1
else:
    stage_fear_again=0
    

st.write(' ##### How many Social Events you attend in a Month?')
social_event=st.number_input('',value=0,key='input3') #max=10

st.write(' ##### How many times you are going outside in a week?')
going_outside=st.number_input('',value=0,key='input4') # max=7

st.write(' ##### Do you feel drained after hanging out with people?')
feel_drain=st.selectbox('',options=['YES','NO'],key='input5')
feel_drain_again=''
if feel_drain=='YES':
    feel_drain_again=1
else:
    feel_drain_again=0

st.write('##### How many friends are there in your circle?')
friend_circle=st.number_input('',value=0,key='input6') #max=15

st.write('##### How many posts do you make on social media on average in a month?')
total_post=st.number_input('',value=0,key='input7') #max=10

predict_button=st.button('CHECK MY PERSONALITY!!')

totest=[time_spend,stage_fear_again,social_event,going_outside,feel_drain_again,friend_circle,total_post]
prediction=dtagain.predict([totest])
if predict_button:
    if prediction==0:
        st.title('YOU ARE AN INTROVERT')
    else:
        st.title('YOU ARE AN EXTROVERT')
