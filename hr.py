import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

train = pd.read_csv('aug_train.csv')
test = pd.read_csv('aug_test.csv')

train = train.drop(columns=['city'])
test = test.drop(columns=['city'])

data = [train,test]
for dataset in data:
    dataset['gender'] = dataset['gender'].fillna('Male')
    gen = {'Male':0,'Female':1,'Other':2}
    dataset['gender'] = dataset['gender'].map(gen)
    dataset['relevent_experience'] = dataset['relevent_experience'].apply(lambda a:a.split(' ')[0])
    exp ={'Has':1,'No':2}
    dataset['relevent_experience'] = dataset['relevent_experience'].map(exp)
    dataset['enrolled_university'] = dataset['enrolled_university'].fillna('no_enrollment')
    enrollment = {'no_enrollment':0, 'Full time course':1, 'Part time course':2}
    dataset['enrolled_university'] = dataset['enrolled_university'].map(enrollment)
    dataset['education_level'] = dataset['education_level'].fillna('Graduate')
    edu = {'Graduate':0, 'Masters':1, 'High School':2, 'Phd':3, 'Primary School':4}
    dataset['education_level'] = dataset['education_level'].map(edu)
    dataset['major_discipline'] = dataset['major_discipline'].fillna('STEM')
    disc = {'STEM':0, 'Humanities':1, 'Other':5, 'Business Degree':2, 'Arts':3, 'No Major':4}
    dataset['major_discipline'] = dataset['major_discipline'].map(disc)
    dataset['experience'] = dataset['experience'].fillna('<1')
    e = {'<1':0, '>20':21,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'11':11,'12':12,'13':13,'14':14,'15':15,'16':16,'17':17,'18':18,'19':19,'20':20}
    dataset['experience'] = dataset['experience'].map(e)
    dataset['company_size'] = dataset['company_size'].fillna(pd.Series(np.random.choice(['50-99','100-500','10000+'], size=len(dataset.index))))
    cs = {'50-99':0,'100-500':1,'10000+':2,'Oct-49':3,'1000-4999':4,'<10':5,'500-999':6,'5000-9999':7}
    dataset['company_size'] = dataset['company_size'].map(cs)
    dataset['company_type'] = dataset['company_type'].fillna('Pvt Ltd')
    typ = {'Pvt Ltd':0,'Funded Startup':1, 'Public Sector':2, 'Early Stage Startup':3, 'NGO':4, 'Other':5}
    dataset['company_type'] = dataset['company_type'].map(typ)
    dataset['last_new_job'] = dataset['last_new_job'].fillna('1')
    la = {'never':0, '1':1,'2':2,'3':3,'4':4,'>4':5}
    dataset['last_new_job'] = dataset['last_new_job'].map(la)
    
test['company_size']=test['company_size'].fillna(3)

y = train.iloc[:,-1]  
X = train.iloc[:,1:-1]
X_test = test.iloc[:,1:]

print(test.isna().sum().sum())
model = sm.OLS(endog=y,exog=X.astype(float)).fit()
print(model.summary())

from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
X = pca.fit_transform(X)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X, y)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


#print(train['gender'].value_counts())
#print(train['enrolled_university'].value_counts())
#print(train['education_level'].value_counts())
#print(train['major_discipline'].value_counts())
#print(train['experience'].value_counts())
#print(train['company_size'].value_counts())
#print(train['company_type'].value_counts())
#print(train['last_new_job'].value_counts())