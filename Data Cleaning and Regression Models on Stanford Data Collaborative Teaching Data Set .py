#!/usr/bin/env python
# coding: utf-8

# In[14]:


# Import the Sanford Data Collaborative Dataset with Pandas 
df2 = pd.read_csv('Sanford_Data_Collaborative_Teaching_DataSet.csv')
df2


# ###### Clean the dataset with Pandas (focus on variables: Age / Sex / ScheduledClinicVisits) 

# In[15]:


# clean data frame column 'Age'
df2['Age'] = df2['Age'].str.replace('+','')
df2


# In[16]:


# clean dataframe column 'sex'
# Check if there are any values that are not male or female 
df2.Sex.unique()


# In[17]:


# remove all the unknown inputs for Sex for a cleaner data 
df2 = df2[df2.Sex != 'Unknown']
# test if they were dropped
df2.Sex.unique()


# In[18]:


# Clean ScheduledClinicVisits
# fill the NA values in ScheduledClinicVisits by a null value 
df2['ScheduledClinicVisits'] = df2.ScheduledClinicVisits.fillna(0)


# ###### Create a scatter plot of age (independent variable, x) and number of scheduled clinic visits (dependent variable, y). Do a separate analysis for males and females. 

# In[19]:


# Scatterplot for Male 
df2_male = df2[df2['Sex']== 'Male']
df2_male[df2_male.Age.apply(lambda x: x.isnumeric())]
df2_male = df2_male.astype({"Age": int})
#plot the scatterplot
plt.scatter(df2_male.Age, df2_male.ScheduledClinicVisits)
plt.xlabel("Age")
plt.ylabel("ScheduledClinicVisits")
plt.title("Male: Age versus ScheduledClinicVisits")


# In[20]:


# Scatterplot for Female  
df2_female = df2[df2['Sex']== 'Female']
df2_female[df2_female.Age.apply(lambda x: x.isnumeric())]
df2_female = df2_female.astype({"Age": int})
#plot the scatterplot
plt.scatter(df2_female.Age, df2_female.ScheduledClinicVisits)
plt.xlabel("Age")
plt.ylabel("ScheduledClinicVisits")
plt.title("Female: Age versus ScheduledClinicVisits")


# ###### Perform a linear regression on age and number of scheduled clinic visits. Do a separate analysis for males and females. 

# In[21]:


# Male linear regression 
y = df2_male.ScheduledClinicVisits  
X = df2_male.Age
X = sm.add_constant(X)
linearR_model = sm.OLS(y, X).fit()
print(linearR_model.summary())
df2_male.corr()


# In[22]:


# Female linear regression 
y = df2_female.ScheduledClinicVisits  
X = df2_female.Age
X = sm.add_constant(X)
linearR_model = sm.OLS(y, X).fit()
print(linearR_model.summary())
df2_female.corr()


# ###### How does number of scheduled clinic visits change with age for males and females (

# From both the scatterploit and the linear regression model that we performed above we can see that per year, a male is going to have 1.348 to 1.714 sheduled clininc visit which will be affected by his age in a directly proprtional manner; the younger the less the scheduled clinic visits. On the other hand per year, a femlae is going to have from 6.140 to 6.585 scheduled clinic visits with a similar direct proprtion with age; the younger the less visits. 

# The results show that despite the similarity in the relationship between age and scheduled clininc visits in both male and female, it shows that there is a big difference in scheduled clininc visits between male and female regardless of age. However, comparing the scatterplot and the linear regression model, it is pretty easy to spot the inaccuracy of a linear regression model to find the relationship between age and scheduled visits. additionally, the r-quared value was pretty low for both male and female, telling us that most of the the variation of our dependent variable can not be predicted by age. The correlation matrices also back this interpretation as they point out that the correlation coefficent between scheduledClinicVisit and Age is 0.166383	for male and 0.019803 for female which both are relatively insignifcant as they are very smaller than either 1 or -1. 

# In[ ]:




