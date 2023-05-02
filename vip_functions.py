#!/usr/bin/env python
# coding: utf-8

# In[11]:


import findspark
findspark.init()
import pyspark
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('demo').getOrCreate()
dfd = spark.sparkContext.textFile("VehicleInsuranceData.csv")
dfs =spark.read.csv("VehicleInsuranceData.csv",inferSchema=True,header=True)
dfp = pd.read_csv("VehicleInsuranceData.csv")

def show_data(num):
    dfp1 = dfp.drop(dfp.index[0])
    df1 = dfp1.head(num)
    print(df1)
    print("\nTotal number of data:",dfs.count())
    print(dfp.info())
    
def show_baic_coverage():
        dfs.select("coverage","Education","EmploymentStatus","Gender","Income","LocationCode","MaritalStatus","Policy","SalesChannel","VehicleClass","VehicleSize").filter((dfs["coverage"]=='Basic')).alias("Basic coverage customers").show()
def show_premium_coverage():
        dfs.select("coverage","Education","EmploymentStatus","Gender","Income","LocationCode","MaritalStatus","Policy","SalesChannel","VehicleClass","VehicleSize").filter((dfs["coverage"]=='Premium')).alias("Premium coverage customers").show()
def show_extended_coverage():
        dfs.select("coverage","Education","EmploymentStatus","Gender","Income","LocationCode","MaritalStatus","Policy","SalesChannel","VehicleClass","VehicleSize").filter((dfs["coverage"]=='Extended')).alias("Extended coverage customers").show()

def show_employee_detail():
        dfs.select("coverage","Education","EmploymentStatus","Gender","Income","LocationCode","MaritalStatus","Policy","SalesChannel","VehicleClass","VehicleSize").filter((dfs["EmploymentStatus"]=='Employed')).alias("Employed customers").show()      
def show_unemployee_detail():
        dfs.select("coverage","Education","EmploymentStatus","Gender","Income","LocationCode","MaritalStatus","Policy","SalesChannel","VehicleClass","VehicleSize").filter((dfs["EmploymentStatus"]=='Unemployed')).alias("Unemployed customers").show()

def show_highschool_detail():
        dfs.select("coverage","Education","EmploymentStatus","Gender","Income","LocationCode","MaritalStatus","Policy","SalesChannel","VehicleClass","VehicleSize").filter((dfs["Education"]=='High School or Below')).alias("Highschool customers").show()
def show_bachelor_detail():
        dfs.select("coverage","Education","EmploymentStatus","Gender","Income","LocationCode","MaritalStatus","Policy","SalesChannel","VehicleClass","VehicleSize").filter((dfs["Education"]=='Bachelor')).alias("Bachelor customers").show()
def show_masters_detail():
        dfs.select("coverage","Education","EmploymentStatus","Gender","Income","LocationCode","MaritalStatus","Policy","SalesChannel","VehicleClass","VehicleSize").filter((dfs["Education"]=='Master')).alias("Master customers").show()

def show_urban_detail():
        dfs.select("coverage","Education","EmploymentStatus","Gender","Income","LocationCode","MaritalStatus","Policy","SalesChannel","VehicleClass","VehicleSize").filter((dfs["LocationCode"]=='Urban')).alias("Urban area customers").show()
def show_suburban_detail():
        dfs.select("coverage","Education","EmploymentStatus","Gender","Income","LocationCode","MaritalStatus","Policy","SalesChannel","VehicleClass","VehicleSize").filter((dfs["LocationCode"]=='Suburban')).alias("Suburban area customers").show()
def show_rural_detail():
        dfs.select("coverage","Education","EmploymentStatus","Gender","Income","LocationCode","MaritalStatus","Policy","SalesChannel","VehicleClass","VehicleSize").filter((dfs["LocationCode"]=='Rural')).alias("Rural area customers").show()

def show_male():
        dfs.select("coverage","Education","EmploymentStatus","Gender","Income","LocationCode","MaritalStatus","Policy","SalesChannel","VehicleClass","VehicleSize").filter((dfs["Gender"]=='M')).alias("Male customers").show()
def show_female():
        dfs.select("coverage","Education","EmploymentStatus","Gender","Income","LocationCode","MaritalStatus","Policy","SalesChannel","VehicleClass","VehicleSize").filter((dfs["Gender"]=='F')).alias("Male customers").show()

def sorting_dataset():
    e=dfd.map(lambda x: x.split(','))
    c=e.map(lambda x: (x[19],x[3])).sortByKey(0).take(10)  
    d=e.map(lambda x: (x[4],x[5])).sortByKey(0).take(10)
    f=e.map(lambda x: (x[6],x[8])).sortByKey(0).take(10)
    h=e.map(lambda x: (x[9],x[10])).sortByKey(0).take(10)
    print("After sorting the data")
    print(c)
    print(d)
    print(f)
    print(h)
    
def decription_dataset():
    des = dfp.describe()
    print(des)
    
def groupby_cov_edu_income():
    grouped = dfp.groupby(['Coverage','Education'])
    result = grouped['Income'].mean()
    print(result)
def groupby_loc_sales_claimamt():
    grouped = dfp.groupby(['LocationCode','SalesChannel'])
    result = grouped['TotalClaimAmount'].mean()
    print(result)
def groupby_pty_ply_vtype():
    grouped = dfp.groupby(['Policy','PolicyType'])
    result = grouped['VehicleSize'].count()
    print(result)
    
def missing_values():
    mis=dfp.isnull().sum()
    print(mis)
    
def categorical_data():
    cov = dfp['Coverage'].value_counts()
    edu = dfp['Education'].value_counts()
    emp = dfp['EmploymentStatus'].value_counts()
    gen = dfp['Gender'].value_counts()
    loc = dfp['LocationCode'].value_counts()
    mar = dfp['MaritalStatus'].value_counts()
    sal = dfp['SalesChannel'].value_counts()
    veh = dfp['VehicleClass'].value_counts()
    print("\nCoverage:",cov)
    print("\nEducation:",edu)
    print("\nEmploymentStatus:",emp)
    print("\nGender:",gen)
    print("\nLocationCode:",loc)
    print("\nMaritalStatus:",mar)
    print("\nSalesChannel:",sal)
    print("\nVehicleClass:",veh)

# Univariate analysis
def income_analysis():
    income = dfp["Income"].describe()
    print(income)
    dfp["Income"].hist(bins=50)
    sns.displot(dfp["Income"])
    print("\n===========================================")
def claim_amount_analysis():
    claim = dfp["TotalClaimAmount"].describe()
    print(claim)
    dfp["TotalClaimAmount"].hist(bins=50)
    sns.displot(dfp["TotalClaimAmount"])
    print("\n===========================================")
def month_inception_analysis():
    month_inception = dfp["MonthsSincePolicyInception"].describe()
    print(month_inception)
    dfp["MonthsSincePolicyInception"].hist(bins=5)
    sns.displot(dfp["MonthsSincePolicyInception"])
    sns.countplot(x="MonthsSincePolicyInception",data = dfp)
    print("\n==============================================")
def gender_analysis():
    gender = dfp["Gender"].describe()
    print(gender)
    print(sns.countplot(x="Gender",data=dfp))
    print("\n==============================================")
def marital_analysis():
    marital = dfp["MaritalStatus"].describe()
    print(marital)
    mm = sns.countplot(x="MaritalStatus",data=dfp)
    print(mm)
    print("\n==============================================")
def coverage_analysis():
    coverage = dfp["Coverage"].describe()
    print(coverage)
    cm=sns.countplot(x="Coverage",data=dfp)
    print(cm)
    print("\n==============================================")
    
# Bivariate analyis
def coverange_monthlypremimum_analysis():
    plt.figure(figsize=(10,5))
    sns.barplot(x="Coverage",y="MonthlyPremiumAuto",data=dfp)
def emp_status_gender_analysis():
    plt.figure(figsize=(10,5))
    sns.barplot(x="EmploymentStatus",y="Income",data=dfp)
def claim_vehicle_analysis():
    plt.figure(figsize=(10,5))
    sns.barplot(x="VehicleSize",y="TotalClaimAmount",data=dfp)
def gender_policy_analysis():
    plt.figure(figsize=(10,5))
    sns.barplot(x="Gender",y="NumberOfPolicies",data=dfp)
def location_clv_analysis():
    plt.figure(figsize=(10,5))
    sns.barplot(x="LocationCode",y="clv",data=dfp)
def heat_map_analysis():
    dfp_drop = dfp.drop(columns=['MonthsSinceLastClaim','MonthsSincePolicyInception','NumberofOpenComplaints'])
    dfp_drop.head()
    plt.figure(figsize=(10,5))
    sns.heatmap(dfp_drop[['clv','Income','MonthlyPremiumAuto','NumberOfPolicies','TotalClaimAmount']].corr(),annot=True)
    plt.show()
    
# outlier analysis
def outlier_analysis():
    plt.figure(figsize=(20,5))
    sns.boxplot(data=dfp)
    
def filter_income():
    dfs.select("EmploymentStatus","Gender","Education").filter((dfs['Income']>=10000)&(dfs['Income']<=45000)).show()
def above_avg_income():
    avg_income = dfp["Income"].mean()
    above_avg_dfp = dfp[dfp["Income"]>avg_income]
    print(above_avg_dfp)
def rich_customers():
    dfs.select("clv","EmploymentStatus","Gender","Education").filter((dfs['Income']>=45000)&(dfs['LocationCode']=='Urban')).show()
def above_avg_claim():
    avg_claim = dfp["TotalClaimAmount"].mean()
    above_avg_claim = dfp[dfp["TotalClaimAmount"]>avg_claim]
    print(above_avg_claim)
def monthly_premium():
    dfs.select("coverage","Education","EmploymentStatus","Gender","Income","LocationCode","MaritalStatus","Policy").filter((dfs['MonthlyPremiumAuto']>=100)).show()
def above_policies():
    dfs.select("coverage","Education","EmploymentStatus","Gender","Income","LocationCode","MaritalStatus","Policy").filter((dfs['NumberOfPolicies']>=8)).show()
def num_of_complaints(n1):
    above_complaints = dfp[dfp["NumberofOpenComplaints"]>n1]
    print(above_complaints)
    
# filter_income()
# above_avg_income()
# rich_customers()
# above_avg_claim()
# monthly_premium()
# above_policies()
# num_of_complaints(43)

        
# outlier_analysis()
    
    
# bivariate
# coverange_monthlypremimum_analysis()
# emp_status_gender_analysis()
# claim_vehicle_analysis()
# gender_policy_analysis()
# location_clv_analysis()
# heat_map_analysis()

# univariate
# income_analysis()    
# claim_amount_analysis()
# month_inception_analysis()
# gender_analysis()
# marital_analysis()
# coverage_analysis()


# categorical_data()     
    
# missing_values()    
    
# decription_dataset()

# show_data(20)
# show_baic_coverage()
# show_premium_coverage()
# show_extended_coverage()
# show_employee_detail()
# show_unemployee_detail()
# show_highschool_detail()
# show_bachelor_detail()
# show_masters_detail()
# show_urban_detail()
# show_suburban_detail()
# show_rural_detail()
# show_male()
# show_female()

# sorting_dataset()

# groupby_cov_edu_income()
# groupby_loc_sales_claimamt()
# groupby_pty_ply_vtype()




# In[ ]:




