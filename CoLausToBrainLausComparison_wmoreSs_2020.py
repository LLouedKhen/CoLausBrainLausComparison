# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 13:36:14 2018

@author: le5988
"""




import os
import numpy as np
import pandas as pd
from pylab import *
import csv 
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import binom
from scipy import stats
from pylab import *
import seaborn as sns
import os
import sys
import matplotlib.mlab as mlab

def cohend(d1,d2):
    n1, n2 = (~np.isnan(d1)).sum(), (~np.isnan(d2)).sum()
    s1, s2 = nanvar(d1, ddof=1), nanvar(d2, ddof = 1)
    s = sqrt(((n1 -1) * s1 + (n2 - 1) * s2) / (n1 + n2 -2))
    u1, u2 = nanmean(d1), nanmean(d2)
    EffectSize = (u1 -u2)/s
    return  EffectSize


allCLData = pd.read_csv('LKhenissi/raw_data_full_CoLaus/CVDrisk_&_cognition_data.csv')
newCLData = pd.read_csv('LKhenissi/Downloads_20181012/Data_AllfromBD/CVDrisk_&_cognition_data.csv')
allCLData.head(20)
allCLData.describe().transpose()

ditchedCols_CLData = allCLData.drop(columns = ['adopt', 'mrtsts2', 'dmst', 'agechd1', 'agechd2', 'brno', 'wnno', 'sprno', 'transferrin_raw'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns = [ 'F1brno', 'F1wnno', 'F1sprno', 'F1handgrip', 'F1lateralite'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns = [ 'F2brno', 'F2wnno', 'F2sprno',  'F2handgrip', 'F2lateralite', 'F2handgrip_com'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['joblvl', 'jobtyp'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['lvlgtx'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['alcusmn', 'alcfrq'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['owt'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['jobact'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['hctld'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['dbtld'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['dbdrg'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['orldrg'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['insn'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['BMI_cat1'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['waist'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['waist_cat1'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['hip'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['F1joblvl'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['F1jobtyp4'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['F1alcpus'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['F1alcusm'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['F1owt'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['F1hctld'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['F1antiDIAB'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['F1BMI_cat1'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['F1waist'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['F1waist_cat1'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['F1hip'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['F2trav1b'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['F2trav1d'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['F2antiHTA'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['F2orldrg'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['F2insn'])
ditchedCols_CLData = ditchedCols_CLData.drop(columns =['F2antiDIAB'])
ditchedCols_CLData= ditchedCols_CLData.drop(columns =['F2alcpus', 'F2alcusm'])
ditchedCols_CLData= ditchedCols_CLData.drop(columns =['F2owt', 'F2BMI_cat1', 'F2waist_cat', ])
ditchedCols_CLData =ditchedCols_CLData.drop(list(ditchedCols_CLData.filter(regex = 'F2ATC')), axis = 1)
ditchedCols_CLData.to_csv('CleanedCLDataa_20201108.csv')
trunCLData = pd.read_csv('CleanedCLDataa_20201108.csv')
trunCLData.head(20)
trunCLData.describe().transpose()

f1f2Date = trunCLData.loc[:,trunCLData.columns.str.contains('datexam')]
dob = trunCLData.loc[:,trunCLData.columns.str.startswith('datbirth')]

#do we need more than one column for DOB? no, but check for nans in first input


anyMissingDOBs = pd.isna(dob).sum()

for i in range (0, len(anyMissingDOBs)):
    if anyMissingDOBs[i] == 0:
        print('Good, do go on.')
        continue
    else:
        print ('Check DOBs')
        
#Now let's work with what Lausanne gave us in terms of DATES
#create dataframe with just dob and exam dates 

dobAndExam = pd.concat([dob, f1f2Date], axis = 1)

#deal with date format (here it is day, month(abbreviated string, abbreviated year)
# but python doesn't like the 20th century so we will oblige by creating a separate array
#and adding '19' to year

x = dob[:]
x = trunCLData['datbirth']
xd = x

nRows = len(dob)

for i in range (0,nRows):
  x[i] = xd.str[:][i][:7] + '19' + xd.str[:][i][7:]
  print (x[i])
  print ('on it', i)
  
x = pd.DataFrame(x)
dobAndExam['datbirth'] = x

#Here we extract dob in correct format
dobAndExam['datbirth'] = pd.to_datetime(dobAndExam['datbirth'], format='%d-%b-%Y')

dobAndExam['F1datexame'] = pd.to_datetime(dobAndExam['F1datexame'], format='%d-%b-%y')

dobAndExam['F2datexam'] = pd.to_datetime(dobAndExam['F2datexam'], format='%d-%b-%y')

#Now compute age and make sure to output in years
dobAndExam['F1Age'] = (dobAndExam['F1datexame'] - dobAndExam['datbirth']).astype('<m8[Y]')

dobAndExam['F2Age'] = (dobAndExam['F2datexam'] - dobAndExam['datbirth']).astype('<m8[Y]')

#Insert these values into othe main data frame, instead of exam date

trunCLData.insert(trunCLData.columns.get_loc('F1datexame'),'F1Age', dobAndExam['F1Age'])
trunCLData.insert(trunCLData.columns.get_loc('F2datexam'),'F2Age', dobAndExam['F2Age'])

#Remove more columns deemed superfluous

trunCLData= trunCLData.drop(columns =['F2datbirth', 'F1datbirth'])

#Now inspect header list  to see what's what, if there are more variables to drop, if vars dicrete or continuous
#which timepoints have what, etc.

varNamesinCurrentCLData = list(trunCLData)

print(varNamesinCurrentCLData)

#identified several values of limited use (see excel sheet for explanations)
#like have you ever been overweight (too many answers missing and also why rely on self-report when we have 
#objective measures)


trunCLData =trunCLData.drop(list(trunCLData.filter(regex = 'sbhs')), axis = 1)

#modified MMSE, modified how? Drop.

trunCLData= trunCLData.drop(columns =['mxsc'])

#lvpl (most live in CH, duh) edtyp (string cat) sclhlp (have F2 data plus unclear what kind of social help) and crbpmed 
#(too many missing)

trunCLData= trunCLData.drop(columns =['lvpl', 'edtyp', 'sclhlp', 'crbpmed'])


#the alcohol variable alcfrq because it has changing units (per day, per week etc). have weekly consumption var

trunCLData =trunCLData.drop(list(trunCLData.filter(regex = 'alcfrq')), axis = 1)

trunCLData.to_csv('CleanedData_20181108.csv')

testingData = trunCLData

# drop F1antiHTA
testingData= testingData.drop(columns =['F1antiHTA'])

#drop F0 diabetes
testingData= testingData.drop(columns =['DIAB'])

#drop F1 diabetes
testingData= testingData.drop(columns =['F1DIAB'])

#Drop following vars for F0
testingData= testingData.drop(columns =['BMI_cat2', 'bmpsc', 'hdlch', 'ldlch', 'trig', 'gluc'])

#Drop following vars for F1
testingData= testingData.drop(columns =['F1BMI_cat2', 'F1bmpsc', 'F1hdlch', 'F1ldlch', 'F1trig', 'F1gluc'])


#Drop following vars for F1
testingData= testingData.drop(columns =['F1crpu', 'F1il1b', 'F1il6', 'F1tnfa'])

#drop alcuse F0, F1; weekly consumption F0, F1; smoking status F0, F1; and F1 HTA

testingData= testingData.drop(columns =['alcuse', 'F1alcuse', 'conso_hebdo', 'F1conso_hebdo', 'sbsmk', 'F1sbsmk', 'F1HTA'])
#Now what are the variables remaining?

varNamesinCurrentCLData = list(testingData)

print(varNamesinCurrentCLData)


testingData.Brainlaus = newCLData.Brainlaus
#Now make one dataframe for CoLaus without BrainLaus and one dataframe for BrainLaus

testingDataCL =testingData.loc[testingData['Brainlaus'] ==0]

testingDataBL = testingData.loc[testingData['Brainlaus'] ==1]

PtsBySex_CL = testingDataCL.groupby('sex').count()
PtsBySex_BL = testingDataBL.groupby('sex').count()

xyCL = [PtsBySex_CL.iat[0,1], PtsBySex_CL.iat[1,1]]
xyCL = np.asarray(xyCL)
xyCL = [xyCL[0], xyCL[1]]

xyBL = [PtsBySex_BL.iat[0,1], PtsBySex_BL.iat[1,1]]
xyBL = np.asarray(xyBL)
xyBL = [xyBL[0], xyBL[1]]

import scipy.stats as stats

print('CoLaus Women and Men: ' + repr(xyCL))
print('BrainLaus Women and Men: ' + repr(xyBL))

#ok here go our first stats; significant differences in females versus males in CL?

genderFreqCL = stats.chisquare(xyCL, [len(testingDataCL)/2, len(testingDataCL)/2])


if genderFreqCL[1] < 0.05:
    print ('The genders are not equally distributed in CoLaus. Chi square yields a p value of ' + repr(genderFreqCL[1]))
else:
    print ('There is no significant difference in the proportion of males and females in the CoLaus cohort')

#Calculate effect size
smChiGenderCL = sm.stats.gof.chisquare_effectsize([len(testingDataCL)/2, len(testingDataCL)/2], xyCL, correction = None, cohen = True, axis = 0)

if smChiGenderCL < 0.2:
    print ('Very small effect size.')
else:
    print ('Look into these differences.')
    
  #Now let's test for differences in gender proportions in the BrainLaus cohort.
genderFreqBL = stats.chisquare(xyBL, [len(testingDataBL)/2, len(testingDataBL)/2])

if genderFreqBL[1] < 0.05:
    print ('The genders are not equally distributed in BrainLaus. Chi square yields a p value of ' + repr(genderFreqBL[1]))
else:
    print ('There is no significant difference in the proportion of males and females in the BrainLaus cohort')

xyCats = ['CoLaus', 'BrainLaus']
xyBar1 = plt.bar([0.8,1.8], [xyCL[0],xyBL[0]], 0.4)
xyBar2 = plt.bar([1.2,2.2], [xyCL[1],xyBL[1]], 0.4)
plt.legend((xyBar1[0], xyBar2[0]), ('Female','Male'))
plt.title('Gender counts in CoLaus and BrainLaus')
plt.xticks([1,2,3,4],xyCats)
plt.show()

  #Now what about the proportion of women to men in CoLaus and BrainLaus?

propGenderCL = xyCL[0]/xyCL[1]

propGenderBL = xyBL[0]/xyBL[1]

propGenderCLBL_Diff = stats.chisquare([propGenderCL, propGenderBL], [len([propGenderCL,propGenderBL])/2, len([propGenderCL, propGenderBL])/2])

if propGenderCLBL_Diff[1] < 0.05:
    print ('The gender proportions are not equivalent in CoLaus vs BrainLaus. Chi square yields a p value of ' + repr(propGenderCLBL_Diff[1]))
else:
    print ('There is no significant difference between CoLaus and BrainLaus proportion of males and females.')
    
#Now check marital status for married, divorced, single and widowed 
    
countMrtstCL = testingDataCL['mrtsts'].value_counts()
countMrtstBL = testingDataBL['mrtsts'].value_counts()
countMrtstCL = countMrtstCL.sort_index(ascending = True)
countMrtstBL = countMrtstBL.sort_index(ascending = True)
propMrtstCL = countMrtstCL/(sum(countMrtstCL))
propMrtstBL = countMrtstBL/(sum(countMrtstBL))
print(countMrtstCL)
print(countMrtstBL)


xyCats = ['Divorced','Married', 'Single', 'Widowed']
xyBar1 = plt.bar([0.8,1.8, 2.8, 3.8], propMrtstCL, 0.4)
xyBar2 = plt.bar([1.2,2.2, 3.2, 4.2], propMrtstBL, 0.4)
plt.legend((xyBar1[0], xyBar2[0]), ('CoLaus', 'BrainLaus'))
plt.title('Marital Status Proportions in CoLaus and BrainLaus')
plt.xticks([1,2,3,4],xyCats)
plt.show()

smChiMrtstCLBL  = stats.chisquare([propMrtstCL, propMrtstBL], [len([propMrtstCL, propMrtstBL])/4, len([propMrtstCL, propMrtstBL])/4, len([propMrtstCL, propMrtstBL])/4, len([propMrtstCL, propMrtstBL])/4])

print(smChiMrtstCLBL[1][:])


status = ['Divorced', 'Married', 'Single', 'Widowed']
for i in range (0, len(smChiMrtstCLBL[1][:])):
    if smChiMrtstCLBL[1][i] < 0.05:
        print('There is a significant difference in  ' + repr(status[i]) +' people between the cohorts.'  )
    else:
        print ('There are no significant differences in the distributions of ' + repr(status[i]) + ' people between CoLaus and BrainLaus')
        
#Is there a significant difference between education types in CoLaus Vs Brain Laus? Chi square, check effect size
#Assume CoLaus proportions are the null; Here, highest education type is coded as 1, middle, 2 and low, 3
        
countEdTyp3CL = testingDataCL['edtyp3'].value_counts()
countEdTyp3BL = testingDataBL['edtyp3'].value_counts()
propEdTyp3CL = countEdTyp3CL/(sum(countEdTyp3CL))
propEdTyp3BL = countEdTyp3BL/(sum(countEdTyp3BL))

#Calculate chi square for each 
diffEdTyp3CL = stats.chisquare(countEdTyp3CL)
diffEdTyp3BL = stats.chisquare(countEdTyp3BL)

if diffEdTyp3CL[1] < 0.05:
    print ('Education types are not equally distributed in CoLaus. Chi square yields a p value of ' + repr(diffEdTyp3CL[1]))
else:
    print ('There is no significant difference in the proportion education types in the CoLaus cohort')

if diffEdTyp3BL[1] < 0.05:
    print ('Education types are not equally distributed in BrainLaus. Chi square yields a p value of ' + repr(diffEdTyp3BL[1]))
else:
    print ('There is no significant difference in the proportion education types in the BrainLaus cohort')
    
yCats = ['Highest Ed','Medium Ed', 'Lowest Ed']
xyBar1 = plt.bar([0.8,1.8, 2.8], sort(propEdTyp3CL), 0.4)
xyBar2 = plt.bar([1.2,2.2, 3.2], sort(propEdTyp3BL), 0.4)
plt.legend((xyBar1[0], xyBar2[0]), ('CoLaus', 'BrainLaus'))
plt.title('Education Type Proportions in CoLaus and BrainLaus')
plt.xticks([1,2,3],xyCats)
plt.show()

smChiEdTypeCLBL  = stats.chisquare([propEdTyp3CL, propEdTyp3BL], [len([propEdTyp3CL, propEdTyp3BL])/3, len([propEdTyp3CL, propEdTyp3BL])/3, len([propEdTyp3CL, propEdTyp3BL])/3])

print ('Education type 1 is highest, 2 is mid-level, 3 is lowest.')
for i in range (0, len(smChiEdTypeCLBL[1][:])):
    if smChiEdTypeCLBL[1][i] < 0.05:
        print('There is a significant difference in proprtion of education type ' + repr(i) )
    else:
        print ('There are no significant differences in the distributions of ' + repr(i) + ' education type between CoLaus and BrainLaus')

d1 = np.array(list(testingDataCL['edlv'].values))
d2 = np.array(list(testingDataBL['edlv'].values))

d1_desc  = testingDataCL['edlv'].describe()
d2_desc  = testingDataBL['edlv'].describe()
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

print(d1_desc, d2_desc)

print('The average education level in years for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus education levels shows a significant difference in the two cohorts.')
    EffectSize = cohend(d1,d2)
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')
else:
    print('There is no significant difference in education levels levels between the two cohorts.')
    
sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('Years of education in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('Education Level (years)')
plt.ylabel('Frequency')


# Are there different proportions of born Swiss in CoLaus and BrainLaus cohorts?
countbrnswsCL = testingDataCL['brnsws'].value_counts()
countbrnswsBL = testingDataBL['brnsws'].value_counts()
propbrnswsCL = countbrnswsCL/(sum(countbrnswsCL))
propbrnswsBL = countbrnswsBL/(sum(countbrnswsBL))

#Calculate chi square for each 
diffbrnswsCL = stats.chisquare(countbrnswsCL)
diffbrnswsBL = stats.chisquare(countbrnswsCL)

xyCats = ['Born Swiss','Foreign Born']
xyBar1 = plt.bar([0.8,1.8], propbrnswsCL, 0.4)
xyBar2 = plt.bar([1.2,2.2], propbrnswsBL, 0.4)
plt.legend((xyBar1[0], xyBar2[0]), ('CoLaus', 'BrainLaus'))
plt.title('Swiss vs Foreign Born Proportions in CoLaus and BrainLaus')
plt.xticks([1,2],xyCats)
plt.show()

smChiBornSwissCLBL  = stats.chisquare([propbrnswsCL, propbrnswsBL], [len([propbrnswsCL, propbrnswsBL])/2, len([propbrnswsCL, propbrnswsBL])/2])


for i in range (0, len(smChiBornSwissCLBL[1][:])):
    if i == 0:
        x = 'Swiss'
    elif i == 1:
        x = 'foreign'
        if smChiBornSwissCLBL[1][i] < 0.05:
            print('There is a significant difference in proprtion of ' + repr(x) + 'born participants.')
        else:
            print ('There are no significant differences in the distributions of Swiss born participants between CoLaus and BrainLaus')
            
#Now check job type at F1 (Job Type 3) which includes low, middle, high status occupation, not working and housewives (5 categories)
countJobTypeCL = testingDataCL['F1jobtyp3'].value_counts()
countJobTypeBL = testingDataBL['F1jobtyp3'].value_counts()

propJobTypeCL = countJobTypeCL/(sum(countJobTypeCL))
propJobTypeBL = countJobTypeBL/(sum(countJobTypeBL))

xyCats = ['High','Middle', 'Low', 'Housewife', 'Not Working']
xyBar1 = plt.bar([0.8,1.8, 2.8, 3.8, 4.8], sort(propJobTypeCL), 0.4)
xyBar2 = plt.bar([1.2,2.2, 3.2, 4.2, 5.2], sort(propJobTypeBL), 0.4)
plt.legend((xyBar1[0], xyBar2[0]), ('CoLaus', 'BrainLaus'))
plt.title('Occupation Type Proportions in CoLaus and BrainLaus, F1')
plt.xticks([1,2,3,4,5],xyCats)
plt.show()

lgVarProp =len([propJobTypeCL, propJobTypeBL])
smChiJobTypeCLBL =stats.chisquare([propJobTypeCL, propJobTypeBL], [lgVarProp/5, lgVarProp/5, lgVarProp/5, lgVarProp/5, lgVarProp/5])

x = ['High level occupation', 'Mid level occupation', 'Low Level occupation', 'Housewife','Not Working']

for i in range (0, len(smChiJobTypeCLBL[1][:])):
    if smChiJobTypeCLBL[1][i] < 0.05:
        print('At F1, There is a significant difference in proportion of ' + repr(x[i]))
    else:
        print ('There are no significant differences in the proportions of ' + repr(x[i]) + ' between CoLaus and BrainLaus at F1.')
        
#Now check for last job type at F1  which includes low, middle, high status occupation, not working (4 cats)

countJobTypeLastCL = testingDataCL['F1joblast'].value_counts()
countJobTypeLastBL = testingDataBL['F1joblast'].value_counts()

propJobTypeLastCL = countJobTypeLastCL/(sum(countJobTypeLastCL))
propJobTypeLastBL = countJobTypeLastBL/(sum(countJobTypeLastBL))

joblCats = ['High', 'Middle', 'Low', 'Not Working']
joblBar1 = plt.bar([0.8,1.8,2.8,3.8], sort(propJobTypeLastCL), 0.4)
joblBar2 = plt.bar([1.2,2.2,3.2,4.2], sort(propJobTypeLastBL), 0.4)
plt.legend((joblBar1[0], joblBar2[0]), ('CoLaus','BrainLaus'))
plt.xticks([1,2,3,4],joblCats)
plt.show()

lgVarProp =len([propJobTypeLastCL, propJobTypeLastBL])
smChiJobTypeLastCLBL =stats.chisquare([propJobTypeLastCL, propJobTypeLastBL], [lgVarProp/4, lgVarProp/4, lgVarProp/4, lgVarProp/4])
print(smChiJobTypeLastCLBL)
print(propJobTypeLastCL)
print(propJobTypeLastBL)

x = ['High level occupation', 'Mid level occupation', 'Low Level occupation', 'Not Working']

for i in range (0, len(smChiJobTypeLastCLBL[1][:])):
    if smChiJobTypeLastCLBL[1][i] < 0.05:
        print('At F1, There is a significant difference in proportion of ' + repr(x[i]) + ' as last known occupation.')
    else:
        print ('There are no significant differences in the proportions of ' + repr(x[i]) + ' as last known occupation between CoLaus and BrainLaus at F1.')
        
#Now, let's look at F2 levels of social help according to retirement and disability. First, retirement (1 is yes, 2 is no)
countF2sclhlp1CL = testingDataCL['F2sclhlp1'].value_counts()
countF2sclhlp1BL = testingDataBL['F2sclhlp1'].value_counts()

propF2sclhlp1CL = countF2sclhlp1CL/(sum(countF2sclhlp1CL))
propF2sclhlp1BL = countF2sclhlp1BL/(sum(countF2sclhlp1BL))

F2sclhlp1Cats = ['Retirement Benefits', 'None', 'Do not know']
F2sclhlp1Bar1 = plt.bar([0.8,1.8, 2.8], propF2sclhlp1CL, 0.4)
                      
F2sclhlp1Bar2 = plt.bar([1.2,2.2, 3.2], propF2sclhlp1BL, 0.4)
                     
plt.legend((F2sclhlp1Bar1[0], F2sclhlp1Bar2[0]), ('CoLaus','BrainLaus'))
plt.title('Proportion of Cohorts Receiving Retirement Benefits, F2')
plt.xticks([1,2,3],F2sclhlp1Cats)
plt.show()

den = len(propF2sclhlp1CL)
varProp = [propF2sclhlp1CL, propF2sclhlp1BL]
lgVarProp =len([propF2sclhlp1CL, propF2sclhlp1BL])
smChiF2sclhlp1CLBL =stats.chisquare(varProp, [lgVarProp/den, lgVarProp/den, lgVarProp/den])

x = ['receiving retirement benefits ', 'receiving no social help','unsure if they are receiving benefits']

for i in range (0, len(smChiF2sclhlp1CLBL[1][:])):
    if smChiF2sclhlp1CLBL[1][i] < 0.05:
        print('At F2, There is a significant difference in proportion of subjects who are ' + repr(x[i]))
    else:
        print ('There are no significant differences in the proportions of ' + repr(x[i]) + ' between CoLaus and BrainLaus at F2.')
        
#Now check for F2 social help (Disability) in CoLaus, Brain Laus
countF2sclhlp2CL =testingDataCL['F2sclhlp2'].value_counts()
countF2sclhlp2CL =countF2sclhlp2CL.sort_index(ascending = True)
countF2sclhlp2BL = testingDataBL['F2sclhlp2'].value_counts()
countF2sclhlp2BL =countF2sclhlp2BL.sort_index(ascending = True)

propF2sclhlp2CL = countF2sclhlp2CL/(sum(countF2sclhlp2CL))
propF2sclhlp2BL = countF2sclhlp2BL/(sum(countF2sclhlp2BL))

F2sclhlp2Cats = ['Disability Benefits', 'None', 'Do not know']
F2sclhlp2Bar1 = plt.bar([0.8,1.8, 2.8], propF2sclhlp2CL, 0.4)
F2sclhlp2Bar2 = plt.bar([1.2,2.2, 3.2], propF2sclhlp2BL, 0.4)
plt.title('Proportion of Cohorts Receiving Disability Benefits, F2')
plt.legend((F2sclhlp2Bar1[0], F2sclhlp2Bar2[0]), ('CoLaus','BrainLaus'))
plt.xticks([1,2,3],F2sclhlp2Cats)
plt.show()

den = len(propF2sclhlp2CL)
varProp = [propF2sclhlp2CL, propF2sclhlp2BL]
lgVarProp =len([propF2sclhlp2CL, propF2sclhlp2BL])
smChiF2sclhlp2CLBL =stats.chisquare(varProp, [lgVarProp/den, lgVarProp/den, lgVarProp/den])

x = ['receiving disability benefits ', 'receiving no social help','unsure if they are receiving benefits']

for i in range (0, len(smChiF2sclhlp2CLBL[1][:])):
    if smChiF2sclhlp2CLBL[1][i] < 0.05:
        print('At F2, There is a significant difference in proportion of subjects who are ' + repr(x[i]))
    else:
        print ('There are no significant differences in the proportions of ' + repr(x[i]) + ' between CoLaus and BrainLaus at F2.')
        
#Now we look at the proportion of people working vs those not working in CoLaus and Brainlaus
countF2trav1CL =testingDataCL['F2trav1'].value_counts()
countF2trav1CL =countF2trav1CL.sort_index(ascending = True)
countF2trav1BL = testingDataBL['F2trav1'].value_counts()
countF2trav1BL =countF2trav1BL.sort_index(ascending = True)

propF2trav1CL = countF2trav1CL/(sum(countF2trav1CL))
propF2trav1BL = countF2trav1BL/(sum(countF2trav1BL))

F2trav1Cats = ['Working', 'Not Working']
F2trav1Bar1 = plt.bar([0.8,1.8], propF2trav1CL, 0.4)
F2trav1Bar2 = plt.bar([1.2,2.2], propF2trav1BL, 0.4)
plt.title('Proportions of working vs non working participants at F2, CoLaus vs BrainLaus')
plt.legend((F2trav1Bar1[0], F2trav1Bar2[0]), ('CoLaus','BrainLaus'))
plt.xticks([1,2],F2trav1Cats)
plt.show()

den = len(propF2trav1CL)
varProp = [propF2trav1CL, propF2trav1BL]
lgVarProp =len(varProp)
smChiF2F2trav1CLBL =stats.chisquare(varProp, [lgVarProp/den, lgVarProp/den])

x = ['working ', 'not working']

for i in range (0, len(smChiF2F2trav1CLBL[1][:])):
    if smChiF2F2trav1CLBL[1][i] < 0.05:
        print('At F2, There is a significant difference in proportion of subjects who are ' + repr(x[i]))
    else:
        print ('There are no significant differences in the proportions of ' + repr(x[i]) + ' between CoLaus and BrainLaus at F2.')
        
#Now, we will look at the proportion of former smokers, non-smokers and smokers in CoLaus and BrainLaus at F2

countF2sbsmkCL = testingDataCL['F2sbsmk'].value_counts()
countF2sbsmkBL = testingDataBL['F2sbsmk'].value_counts()
propF2sbsmkCL = countF2sbsmkCL/(sum(countF2sbsmkCL))
propF2sbsmkBL = countF2sbsmkBL/(sum(countF2sbsmkBL))

F2sbsmkCats = ['Former', 'Non-Smoker', 'Smoker']
F2sbsmkBar1 = plt.bar([0.8, 1.8, 2.8], propF2sbsmkCL, 0.4)
F2sbsmkBar2 = plt.bar([1.2, 2.2, 3.2], propF2sbsmkBL, 0.4)
plt.title('Proportions of former smokers, non smokers and smokers at F2, CoLaus vs BrainLaus')
plt.legend((F2sbsmkBar1[0], F2sbsmkBar2[0]), ('CoLaus','BrainLaus'))
plt.xticks([1,2,3],F2sbsmkCats)
plt.show()

den = len(propF2sbsmkCL)
varProp = [propF2sbsmkCL, propF2sbsmkBL]
lgVarProp =len(varProp)
smChiF2F2sbsmkCLBL =stats.chisquare(varProp, [lgVarProp/den, lgVarProp/den, lgVarProp/den])

x = ['former smokers', 'non-smokers', 'smokers']

for i in range (0, len(smChiF2F2sbsmkCLBL[1][:])):
    if smChiF2F2sbsmkCLBL[1][i] < 0.05:
        print('At F2, There is a significant difference in proportion of subjects who are ' + repr(x[i]))
    else:
        print ('There are no significant differences in the proportions of ' + repr(x[i]) + ' between CoLaus and BrainLaus at F2.')

#Now check for proportion of diabetics in CoLaus, Brain Laus
        
countF2DIABCL =testingDataCL['F2DIAB'].value_counts()
countF2DIABBL = testingDataBL['F2DIAB'].value_counts()
propF2DIABCL = countF2DIABCL/(sum(countF2DIABCL))
propF2DIABBL = countF2DIABBL/(sum(countF2DIABBL))

F2DIABCats = ['Not diabetic', 'Diabetic']
F2DIABBar1 = plt.bar([0.8,1.8], propF2DIABCL, 0.4)
F2DIABBar2 = plt.bar([1.2,2.2], propF2DIABBL, 0.4)
plt.title('Proportions of  non-diabetics and diabetics at F2, CoLaus vs BrainLaus')
plt.legend((F2DIABBar1[0], F2DIABBar2[0]), ('CoLaus','BrainLaus'))
plt.xticks([1,2],F2DIABCats)
plt.show()

den = len(propF2DIABCL)
varProp = [propF2DIABCL, propF2DIABBL]
lgVarProp =len(varProp)
smChiF2DIABCLBL =stats.chisquare(varProp, [lgVarProp/den, lgVarProp/den])

x = ['non diabetics', 'diabetics']

for i in range (0, len(smChiF2DIABCLBL[1][:])):
    if smChiF2DIABCLBL[1][i] < 0.05:
        print('At F2, There is a significant difference in proportion of subjects who are ' + repr(x[i]))
    else:
        print ('There are no significant differences in the proportions of ' + repr(x[i]) + ' between CoLaus and BrainLaus at F2.')
        
#Now, check for proportion of participants diagnosed with hypertension in CoLaus, BrainLaus
        
countF2HTACL =testingDataCL['F2HTA'].value_counts()
countF2HTACL =countF2HTACL.sort_index(ascending = True)
countF2HTABL = testingDataBL['F2HTA'].value_counts()
countF2HTABL =countF2HTABL.sort_index(ascending = True)
propF2HTACL = countF2HTACL/(sum(countF2HTACL))
propF2HTABL = countF2HTABL/(sum(countF2HTABL))

F2HTACats = ['None', 'Hypertension']
F2HTABar1 = plt.bar([0.8,1.8], propF2HTACL, 0.4)
F2HTABar2 = plt.bar([1.2,2.2], propF2HTABL, 0.4)
plt.title('Proportion of  hypertensives at F2, CoLaus vs BrainLaus')
plt.legend((F2HTABar1[0], F2HTABar2[0]), ('CoLaus','BrainLaus'))
plt.xticks([1,2],F2HTACats)
plt.show()

den = len(propF2HTACL)
varProp = [propF2HTACL, propF2HTABL]
lgVarProp =len(varProp)
smChiF2HTACLBL =stats.chisquare(varProp, [lgVarProp/den, lgVarProp/den])

x = ['non hypertensives', 'hypertensives']

for i in range (0, len(smChiF2HTACLBL[1][:])):
    if smChiF2HTACLBL[1][i] < 0.05:
        print('At F2, There is a significant difference in proportion of subjects who are ' + repr(x[i]))
    else:
        print ('There are no significant differences in the proportions of ' + repr(x[i]) + ' between CoLaus and BrainLaus at F2.')

countF2cmpCL =testingDataCL['F2cmp'].value_counts()
countF2cmpCL =countF2cmpCL.sort_index(ascending = True)
countF1cmpCL =testingDataCL['F1cmp'].value_counts()
countF1cmpCL =countF1cmpCL.sort_index(ascending = True)
countF2cmpBL = testingDataBL['F2cmp'].value_counts()
countF2cmpBL =countF2cmpBL.sort_index(ascending = True)
countF1cmpBL = testingDataBL['F1cmp'].value_counts()
countF1cmpBL =countF1cmpBL.sort_index(ascending = True)
countbyF2cmpCL = countF2cmpCL + countF1cmpCL
countbyF2cmpBL = countF2cmpBL + countF1cmpBL

#drop the nan column 
countbyF2cmpCL = countbyF2cmpCL.drop(labels =9)
countbyF2cmpBL = countbyF2cmpBL.drop(labels =9)
countF2cmpCL = countF2cmpCL.drop(labels =9)
countF2cmpBL = countF2cmpBL.drop(labels =9)

propbyF2cmpCL = countbyF2cmpCL/(sum(countF2cmpCL) + sum(countF1cmpCL))
propbyF2cmpBL = countbyF2cmpBL/(sum(countF2cmpBL) + sum(countF1cmpBL))

F2cmpCats = ['Cardiomyopathy', 'No cardiomyopathy']
F2cmpBar1 = plt.bar([0.8,1.8], propbyF2cmpCL, 0.4)
F2cmpBar2 = plt.bar([1.2,2.2], propbyF2cmpBL, 0.4)
plt.title('Proportions of cardiomyopathy events by F2, CoLaus vs BrainLaus')
plt.legend((F2cmpBar1[0], F2cmpBar2[0]), ('CoLaus','BrainLaus'))
plt.xticks([1,2],F2cmpCats)
plt.show()

den = len(propbyF2cmpCL)
varProp = [propbyF2cmpCL, propbyF2cmpBL]
lgVarProp =len(varProp)
smChibyF2cmpCLBL =stats.chisquare(varProp, [lgVarProp/den, lgVarProp/den])

x = ['a cardiomyopathy', 'no cardiomyopathy']

for i in range (0, len(smChibyF2cmpCLBL[1][:])):
    if smChibyF2cmpCLBL[1][i] < 0.05:
        print('At F2, There is a significant difference in proportion of subjects who have had ' + repr(x[i]))
    else:
        print ('There are no significant differences in the proportions of ' + repr(x[i]) + ' between CoLaus and BrainLaus at F2.')
        
#Now look at CVR event valvular heart disease (hdv)

countF2hdvCL =testingDataCL['F2hdv'].value_counts()
countF2hdvCL =countF2hdvCL.sort_index(ascending = True)
countF1hdvCL =testingDataCL['F1hdv'].value_counts()
countF1hdvCL =countF1hdvCL.sort_index(ascending = True)
countF2hdvBL = testingDataBL['F2hdv'].value_counts()
countF2hdvBL =countF2hdvBL.sort_index(ascending = True)
countF1hdvBL = testingDataBL['F1hdv'].value_counts()
countF1hdvBL =countF1hdvBL.sort_index(ascending = True)
countbyF2hdvCL = countF2hdvCL + countF1hdvCL
countbyF2hdvBL = countF2hdvBL + countF1hdvBL

#drop the nan column 
countbyF2hdvCL = countbyF2hdvCL.drop(labels =9)
countbyF2hdvBL = countbyF2hdvBL.drop(labels =9)
countF2hdvCL = countF2hdvCL.drop(labels =9)
countF2hdvBL = countF2hdvBL.drop(labels =9)

propbyF2hdvCL = countbyF2hdvCL/(sum(countF2hdvCL) + sum(countF1hdvCL))
propbyF2hdvBL = countbyF2hdvBL/(sum(countF2hdvBL) + sum(countF1hdvBL))

F2hdvCats = ['Valvular heart disease', 'None']
F2hdvBar1 = plt.bar([0.8,1.8], propbyF2hdvCL, 0.4)
F2hdvBar2 = plt.bar([1.2,2.2], propbyF2hdvBL, 0.4)
plt.title('Proportions of valvular heart disease events by F2, CoLaus vs BrainLaus')
plt.legend((F2hdvBar1[0], F2hdvBar2[0]), ('CoLaus','BrainLaus'))
plt.xticks([1,2],F2hdvCats)
plt.show()

den = len(propbyF2hdvCL)
varProp = [propbyF2hdvCL, propbyF2hdvBL]
lgVarProp =len(varProp)
smChibyF2hdvCLBL =stats.chisquare(varProp, [lgVarProp/den, lgVarProp/den])

x = ['valcular heart disease', 'no valvular heart disease']

for i in range (0, len(smChibyF2hdvCLBL[1][:])):
    if smChibyF2hdvCLBL[1][i] < 0.05:
        print('At F2, There is a significant difference in proportion of subjects who have had ' + repr(x[i]))
    else:
        print ('There are no significant differences in the proportions of ' + repr(x[i]) + ' between CoLaus and BrainLaus at F2.')
        
 #Now check for proportion of heart failure (chf) in CoLaus, Brain Laus

countF2chfCL =testingDataCL['F2chf'].value_counts()
countF2chfCL =countF2chfCL.sort_index(ascending = True)
countF1chfCL =testingDataCL['F1chf'].value_counts()
countF1chfCL =countF1chfCL.sort_index(ascending = True)
countF2chfBL = testingDataBL['F2chf'].value_counts()
countF2chfBL =countF2chfBL.sort_index(ascending = True)
countF1chfBL = testingDataBL['F1chf'].value_counts()
countF1chfBL =countF1chfBL.sort_index(ascending = True)
countbyF2chfCL = countF2chfCL + countF1chfCL
countbyF2chfBL = countF2chfBL + countF1chfBL


#drop the nan column 
countbyF2chfCL = countbyF2chfCL.drop(labels =9)
countbyF2chfBL = countbyF2chfBL.drop(labels =9)
countF2chfCL = countF2chfCL.drop(labels =9)
countF2chfBL = countF2chfBL.drop(labels =9)

propbyF2chfCL = countbyF2chfCL/(sum(countF2chfCL) + sum(countF1chfCL))
propbyF2chfBL = countbyF2chfBL/(sum(countF2chfBL) + sum(countF1chfBL))

F2chfCats = ['Heart Failure', 'None']
F2chfBar1 = plt.bar([0.8,1.8], propbyF2chfCL, 0.4)
F2chfBar2 = plt.bar([1.2,2.2], propbyF2chfBL, 0.4)
plt.title('Proportion of heart failure by F2, CoLaus vs BrainLaus')
plt.legend((F2chfBar1[0], F2chfBar2[0]), ('CoLaus','BrainLaus'))
plt.xticks([1,2],F2chfCats)
plt.show()

den = len(propbyF2chfCL)
varProp = [propbyF2chfCL, propbyF2chfBL]
lgVarProp =len(varProp)
smChibyF2chfCLBL =stats.chisquare(varProp, [lgVarProp/den, lgVarProp/den])

x = ['heart failure', 'no heart failure']

for i in range (0, len(smChibyF2chfCLBL[1][:])):
    if smChibyF2chfCLBL[1][i] < 0.05:
        print('At F2, There is a significant difference in proportion of subjects who have had ' + repr(x[i]))
    else:
        print ('There are no significant differences in the proportions of ' + repr(x[i]) + ' between CoLaus and BrainLaus at F2.')
        
 #Now check for proportion of arrythmia in CoLaus, Brain Laus
 #NB here I am taking the sum of yes events for F1 and F2

countF2artmCL =testingDataCL['F2artm'].value_counts()
countF2artmCL =countF2artmCL.sort_index(ascending = True)
countF1artmCL =testingDataCL['F1artm'].value_counts()
countF1artmCL =countF1artmCL.sort_index(ascending = True)
countF2artmBL = testingDataBL['F2artm'].value_counts()
countF2artmBL =countF2artmBL.sort_index(ascending = True)
countF1artmBL = testingDataBL['F1artm'].value_counts()
countF1artmBL =countF1artmBL.sort_index(ascending = True)
countbyF2artmCL = countF2artmCL + countF1artmCL
countbyF2artmBL = countF2artmBL + countF1artmBL

#drop the nan column 
countbyF2artmCL = countbyF2artmCL.drop(labels =9)
countbyF2artmBL = countbyF2artmBL.drop(labels =9)
countF2artmCL = countF2artmCL.drop(labels =9)
countF2artmBL = countF2artmBL.drop(labels =9)

propbyF2artmCL = countbyF2artmCL/(sum(countF2artmCL) + sum(countF1artmCL))
propbyF2artmBL = countbyF2artmBL/(sum(countF2artmBL) + sum(countF1artmBL))

F2artmCats = ['Arrythmia', 'None']
F2artmBar1 = plt.bar([0.8,1.8], propbyF2artmCL, 0.4)
F2artmBar2 = plt.bar([1.2,2.2], propbyF2artmBL, 0.4)
plt.title('Proportion of arrythmia events by F2, CoLaus vs BrainLaus')
plt.legend((F2artmBar1[0], F2artmBar2[0]), ('CoLaus','BrainLaus'))
plt.xticks([1,2],F2artmCats)
plt.show()

den = len(propbyF2artmCL)
varProp = [propbyF2artmCL, propbyF2artmBL]
lgVarProp =len(varProp)
smChibyF2artmCLBL =stats.chisquare(varProp, [lgVarProp/den, lgVarProp/den])

x = ['arrythmia', 'no arrythmia']

for i in range (0, len(smChibyF2artmCLBL[1][:])):
    if smChibyF2artmCLBL[1][i] < 0.05:
        print('At F2, There is a significant difference in proportion of subjects who have had ' + repr(x[i]))
    else:
        print ('There are no significant differences in the proportions of ' + repr(x[i]) + ' between CoLaus and BrainLaus at F2.')
        

#Now check for proportion of angina (angn) in CoLaus, Brain Laus
 #NB here there only seem to be data for F2
      
countF2angnCL =testingDataCL['F2angn'].value_counts()
countF2angnCL =countF2angnCL.sort_index(ascending = True)
countF2angnBL = testingDataBL['F2angn'].value_counts()
countF2angnBL =countF2angnBL.sort_index(ascending = True)

#drop the nan column 
countF2angnCL = countF2angnCL.drop(labels =9)
countF2angnBL = countF2angnBL.drop(labels =9)

propbyF2angnCL = countF2angnCL/(sum(countF2angnCL))
propbyF2angnBL = countF2angnBL/(sum(countF2angnBL))


F2angnCats = ['Angina', 'None']
F2angnBar1 = plt.bar([0.8,1.8], propbyF2angnCL, 0.4)
F2angnBar2 = plt.bar([1.2,2.2], propbyF2angnBL, 0.4)
plt.title('Proportion of angina events by F2, CoLaus vs BrainLaus')
plt.legend((F2angnBar1[0], F2angnBar2[0]), ('CoLaus','BrainLaus'))
plt.xticks([1,2],F2angnCats)
plt.show()

den = len(propbyF2angnCL)
varProp = [propbyF2angnCL, propbyF2angnBL]
lgVarProp =len(varProp)
smChibyF2angnCLBL =stats.chisquare(varProp, [lgVarProp/den, lgVarProp/den])

x = ['angina', 'no angina']

for i in range (0, len(smChibyF2angnCLBL[1][:])):
    if smChibyF2angnCLBL[1][i] < 0.05:
        print('At F2, There is a significant difference in proportion of subjects who have had ' + repr(x[i]))
    else:
        print ('There are no significant differences in the proportions of ' + repr(x[i]) + ' between CoLaus and BrainLaus at F2.')

 #Now check for proportion of miocardial infarction in CoLaus, Brain Laus
 #NB here I am using F1 + f2 data
 
countF2miacCL =testingDataCL['F2miac'].value_counts()
countF2miacCL =countF2miacCL.sort_index(ascending = True)
countF1miacCL =testingDataCL['F1miac'].value_counts()
countF1miacCL =countF1miacCL.sort_index(ascending = True)

countF2miacBL = testingDataBL['F2miac'].value_counts()
countF2miacBL =countF2miacBL.sort_index(ascending = True)
countF1miacBL = testingDataBL['F1miac'].value_counts()
countF1miacBL =countF1miacBL.sort_index(ascending = True)

countbyF2miacCL = countF2miacCL + countF1miacCL
countbyF2miacBL = countF2miacBL + countF1miacBL

#drop the nan column 
countbyF2miacCL = countbyF2miacCL.drop(labels =9)
countbyF2miacBL = countbyF2miacBL.drop(labels =9)
countF2miacCL = countF2miacCL.drop(labels =9)
countF2miacBL = countF2miacBL.drop(labels =9)

propbyF2miacCL = countbyF2miacCL/(sum(countF2miacCL) + sum(countF1miacCL))
propbyF2miacBL = countbyF2miacBL/(sum(countF2miacBL) + sum(countF1miacBL))

F2miacCats = ['Miocardial Infarction', 'None']
F2miacBar1 = plt.bar([0.8,1.8], propbyF2miacCL, 0.4)
F2miacBar2 = plt.bar([1.2,2.2], propbyF2miacBL, 0.4)
plt.title('Proportion of myiocardial infarction events by F2, CoLaus vs BrainLaus')
plt.legend((F2miacBar1[0], F2miacBar2[0]), ('CoLaus','BrainLaus'))
plt.xticks([1,2],F2miacCats)
plt.show()

den = len(propbyF2miacCL)
varProp = [propbyF2miacCL, propbyF2miacBL]
lgVarProp =len(varProp)
smChibyF2miacCLBL =stats.chisquare(varProp, [lgVarProp/den, lgVarProp/den])

x = ['miocardial infarction', 'no miocardial infarction']

for i in range (0, len(smChibyF2miacCLBL[1][:])):
    if smChibyF2miacCLBL[1][i] < 0.05:
        print('At F2, There is a significant difference in proportion of subjects who have had ' + repr(x[i]))
    else:
        print ('There are no significant differences in the proportions of ' + repr(x[i]) + ' between CoLaus and BrainLaus at F2.')
        
#Finally, I will sum all CVR indicators above and compare events in CoLaus vs BrainLaus
countbyF2CardCL = countF2angnCL + countbyF2artmCL + countbyF2chfCL + countbyF2cmpCL + countbyF2hdvCL + countbyF2miacCL
countbyF2CardBL = countF2angnBL + countbyF2artmBL + countbyF2chfBL + countbyF2cmpBL + countbyF2hdvBL + countbyF2miacBL

propbyF2CardCL = countbyF2CardCL/(sum(countbyF2CardCL))
propbyF2CardBL = countbyF2CardBL/(sum(countbyF2CardBL))

F2CardCats = ['Cardiac Event', 'None']
F2CardBar1 = plt.bar([0.8,1.8], propbyF2CardCL, 0.4)
F2CardBar2 = plt.bar([1.2,2.2], propbyF2CardBL, 0.4)
plt.title('Proportion of all cardiac events by F2, CoLaus vs BrainLaus')
plt.legend((F2CardBar1[0], F2CardBar2[0]), ('CoLaus','BrainLaus'))
plt.xticks([1,2],F2CardCats)
plt.show()

den = len(propbyF2CardCL)
varProp = [propbyF2CardCL, propbyF2CardBL]
lgVarProp =len(varProp)
smChibyF2CardCLBL =stats.chisquare(varProp, [lgVarProp/den, lgVarProp/den])

x = ['cardiac events', 'no cardiac events']

for i in range (0, len(smChibyF2CardCLBL[1][:])):
    if smChibyF2CardCLBL[1][i] < 0.05:
        print('At F2, There is a significant difference in proportion of subjects who have had ' + repr(x[i]))
    else:
        print ('There are no significant differences in the proportions of ' + repr(x[i]) + ' between CoLaus and BrainLaus at F2.')

#We will now move on to testing continuous variables in CoLaus and BrainLaus cohorts



#We will first look at age distributions in the two cohorts

ageCL = np.array(list(testingDataCL['F2Age'].values))
ageBL = np.array(list(testingDataBL['F2Age'].values))
d1 = ageCL
d2 = ageBL

ageCL_desc  = testingDataCL['F2Age'].describe()
print(ageCL_desc)

ageBL_desc  = testingDataBL['F2Age'].describe()
print(ageBL_desc)

meanCL = np.nanmean(ageCL)
stdCL =np.nanstd(ageCL)
meanBL = np.nanmean(ageBL)
stdBL =np.nanstd(ageBL)

print('The average age of subjects in the CoLaus(non-brain) cohort at F2 is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('Age Distributions in CoLaus and BrainLaus')
plt.legend(loc ='upper right')
plt.xlabel('Age')
plt.ylabel('Frequency')

diffAgeTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')


if diffAgeTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus cohort age distributions shows a significant difference in the two cohorts, with the BrainLaus cohort being younger')
    #compute effect size
    EffectSize = cohend(d1,d2)
    if EffectSize > 0.2:
        print('The effect size for age is ' +repr(EffectSize))
    else:
        print('The effect size for age is ' +repr(EffectSize))
        print('The effect size is too small to be meaningful.')   
else:
    print('There is no significant difference in age between the two cohorts.')
    
#We now test for number of children begotten by the different cohorts
nochdCL = np.array(list(testingDataCL['nochd'].values))
nochdBL = np.array(list(testingDataBL['nochd'].values))
d1 = nochdCL
d2 = nochdBL
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

nochdCL_desc  = testingDataCL.nochd.describe()
nochdBL_desc  = testingDataBL.nochd.describe()
print(nochdCL_desc, nochdBL_desc )

print('The average number of children for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffnochdTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffnochdTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus number of children shows a significant difference in the two cohorts, with the BrainLaus cohort being younger')
    EffectSize = cohend(d1,d2)
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + 'which means there is a considerable difference between the two groups.')
    else:
        print('The effect size ' +repr (EffectSize) + ' is too small to be meaningful.')   
else:
    print('There is no significant difference in number of children between the two cohorts.')

sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('Number of children in CoLaus and BrainLaus Participant Households')
plt.legend(loc ='upper right')
plt.xlabel('Number of children')
plt.ylabel('Frequency')

#Now let's look at differences in minutes walked (to work) between the two cohorts
mnwlkCL = np.array(list(testingDataCL['mnwlk'].values))
mnwlkBL = np.array(list(testingDataBL['mnwlk'].values))
d1 = mnwlkCL
d2 = mnwlkBL
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

d1_desc  = testingDataCL['mnwlk'].describe()
d2_desc  = testingDataBL['mnwlk'].describe()
print(d1_desc,
      d2_desc)

print('The average number of minutes walked for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffmnwlkTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffmnwlkTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus minutes walked to work shows a significant difference in the two cohorts, with the nonBrainLaus cohort walking more to get to work.')
    EffectSize = cohend(d1,d2)
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + 'which means there is a considerable difference between the two groups.')
    else:
        print('The effect size  ' +repr (EffectSize) + '  is too small to be meaningful.')   
else:
    print('There is no significant difference in number of minutes walked between the two cohorts.')
    
sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('Number of minutes walked in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('Number of minutes walked')
plt.ylabel('Frequency')


countphyactCL = testingDataCL['phyact'].value_counts()
countphyactCL = countphyactCL.sort_index(ascending = True)
countphyactBL = testingDataBL['phyact'].value_counts()
countphyactBL = countphyactBL.sort_index(ascending = True)
d1 = countphyactCL
d2 = countphyactBL

propphyactCL = countphyactCL/(sum(countphyactCL))
propphyactBL = countphyactBL/(sum(countphyactBL))

propphyactCL=np.append(propphyactCL, 0)
a = propphyactCL[2] 
b = propphyactCL[3]
c = propphyactCL[4]
propphyactCL[2] = c
propphyactCL[3] = a
propphyactCL[4] = b

propphyactBL=np.append(propphyactBL, 0)
a = propphyactBL[2] 
b = propphyactBL[3]
c = propphyactBL[4]
propphyactBL[2] = c
propphyactBL[3] = a
propphyactBL[4] = b
print(propphyactBL)
print(propphyactCL)

propphyactCL = propphyactCL[:5]
propphyactBL = propphyactBL[:5]

phyactCats = ['1/week', '2/week', '>3/week', 'Do not know', 'No exercise']
phyactBar1 = plt.bar([0.8,1.8, 2.8, 3.8, 4.8], propphyactCL, 0.4)
phyactBar2 = plt.bar([1.2,2.2, 3.2, 4.2, 5.2], propphyactBL, 0.4)
plt.title('Proportion of people engaging in different levels of exercis, CoLaus vs BrainLaus')
plt.legend((phyactBar1[0], phyactBar2[0]), ('CoLaus','BrainLaus'))
plt.xticks([1, 2, 3, 4, 5],phyactCats)
plt.show()

den = len(propphyactCL)
varProp = [propphyactCL, propphyactBL]
lgVarProp =len(varProp)
smChiphyactCLBL =stats.chisquare(varProp, [lgVarProp/den, lgVarProp/den,  lgVarProp/den,  lgVarProp/den,  lgVarProp/den])

x = ['1/week', '2/week', '>3/week', 'Do not know', 'No exercise']

for i in range (0, len(smChiphyactCLBL[1][:])):
    if smChiphyactCLBL[1][i] < 0.05:
        print('At F2, There is a significant difference in proportion of subjects who exercise ' + repr(x[i]))
    else:
        print ('There are no significant differences in the proportions of people who exercise ' + repr(x[i]) + ' between CoLaus and BrainLaus at F2.')

# Now we will look at several blood markers, starting with ferritin
        
ferritinCL = np.array(list(testingDataCL['ferritin'].values))
ferritinBL = np.array(list(testingDataBL['ferritin'].values))
d1 = ferritinCL
d2 = ferritinBL
d1 = np.log10(d1)
d2 = np.log10(d2)
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

d1_desc  = testingDataCL['ferritin'].describe()
d2_desc  = testingDataBL['ferritin'].describe()
print(d1_desc, d2_desc)

print('The average ferritin count for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffferritinTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffferritinTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus ferritin levels shows a significant difference in the two cohorts, with higher levels in the CoLaus cohort.')
    EffectSize = cohend(d1,d2)
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')   
else:
    print('There is no significant difference in ferritin levels between the two cohorts.')
    
sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('Ferritin count in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('Ferritin count')
plt.ylabel('Frequency')

transferrinCL = np.array(list(testingDataCL['transferrin'].values))
transferrinBL = np.array(list(testingDataBL['transferrin'].values))
d1 = transferrinCL
d2 = transferrinBL
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

d1_desc  = testingDataCL['transferrin'].describe()
d2_desc  = testingDataBL['transferrin'].describe()
print(d1_desc, d2_desc)

print('The average transferrin count for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

difftransferrinTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if difftransferrinTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus transferrin levels shows a significant difference in the two cohorts, with the nonBrainLaus cohort walking more to get to work.')
    EffectSize = cohend(d1,d2)
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')   
else:
    print('There is no significant difference in ferritin levels between the two cohorts.')
    
sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('Transferrin count in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('Transferrin count')
plt.ylabel('Frequency')

F1insulinCL = np.array(list(testingDataCL['F1insulin'].values))
F1insulinBL = np.array(list(testingDataBL['F1insulin'].values))
d1 = F1insulinCL
d2 = F1insulinBL
d1 = np.log10(d1)
d2 = np.log10(d2)
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

d1_desc  = testingDataCL['F1insulin'].describe()
d2_desc  = testingDataBL['F1insulin'].describe()
print(d1_desc, d2_desc)

print('The average F1insulin level for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffF1insulinTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffF1insulinTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus F1insulin levels shows a significant difference in the two cohorts, with the CoLaus cohort showing higher insulin levels.')
    EffectSize = cohend(d1,d2)
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')   
else:
    print('There is no significant difference in F1insulin levels between the two cohorts.')

sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('Insulin Levels in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('Insulin')
plt.ylabel('Frequency')

#Is there a difference between non-BrainLaus and BrainLaus cohorts in adiponectin levels?

F1adiponectinCL = np.array(list(testingDataCL['F1adiponectin'].values))
F1adiponectinBL = np.array(list(testingDataBL['F1adiponectin'].values))
d1 = F1adiponectinCL
d2 = F1adiponectinBL
d1 = np.log10(d1)
d2 = np.log10(d2)
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

d1_desc  = testingDataCL['F1adiponectin'].describe()
d2_desc  = testingDataBL['F1adiponectin'].describe()
print(d1_desc, d2_desc)

print('The average F1adiponectin level for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffF1adiponectinTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffF1adiponectinTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus F1adiponectin levels shows a significant difference in the two cohorts.')
    EffectSize = cohend(d1,d2)
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')   
else:
    print('There is no significant difference in F1adiponectin levels between the two cohorts.')
    
sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('F1 adiponectin Levels in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('F1 adiponectin')
plt.ylabel('Frequency')

d1 = np.array(list(testingDataCL['F1leptin'].values))
d2 = np.array(list(testingDataBL['F1leptin'].values))
d1 = np.log10(d1)
d2 = np.log10(d2)
d1_desc  = testingDataCL['F1leptin'].describe()
d2_desc  = testingDataBL['F1leptin'].describe()
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

print(d1_desc, d2_desc)

print('The average F1leptin level for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffF1leptinTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffF1leptinTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus F1leptin levels shows a significant difference in the two cohorts,  with CoLaus subjects having higher leptin levels.')
    EffectSize = cohend(d1,d2)
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')   
else:
    print('There is no significant difference in F1leptin levels between the two cohorts.')
    
sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('F1 leptin Levels in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('F1 leptin')
plt.ylabel('Frequency')

#Is there a difference in average alcohol consumption between the two cohorts?

d1 = np.array(list(testingDataCL['F2conso_hebdo'].values))
d2 = np.array(list(testingDataBL['F2conso_hebdo'].values))
d1_desc  = testingDataCL['F2conso_hebdo'].describe()
d2_desc  = testingDataBL['F2conso_hebdo'].describe()
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

print(d1_desc, d2_desc)

print('The average F2conso_hebdo level for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffF2conso_hebdoTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffF2conso_hebdoTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus F2conso_hebdo levels shows a significant difference in the two cohorts, with BrainLaus Subjects drinking more.')
    EffectSize = cohend(d1,d2)
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')
else:
    print('There is no significant difference in F2conso_hebdo levels between the two cohorts.')

sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('F2 weekly alcohol consumption in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('F2 weekly alcohol consumption (Units)')
plt.ylabel('Frequency')

#Are there differences in participants' weight between CoLaus and Brain Laus?

d1 = np.array(list(testingDataCL['F2wt'].values))
d2 = np.array(list(testingDataBL['F2wt'].values))
d1_desc  = testingDataCL['F2wt'].describe()
d2_desc  = testingDataBL['F2wt'].describe()
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

print(d1_desc, d2_desc)

print('The average F2wt level for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffF2wtTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffF2wtTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus F2wt levels shows a significant difference in the two cohorts.')
    EffectSize = cohend(d1,d2)
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')
else:
    print('There is no significant difference in F2wt levels between the two cohorts.')
    
sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('F2 weight in kg in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('F2 weight (kg)')
plt.ylabel('Frequency')

#What about BMI?

d1 = np.array(list(testingDataCL['F2BMI'].values))
d2 = np.array(list(testingDataBL['F2BMI'].values))
d1_desc  = testingDataCL['F2BMI'].describe()
d2_desc  = testingDataBL['F2BMI'].describe()
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

print(d1_desc, d2_desc)

print('The average F2 BMI for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffF2BMITtest =stats.ttest_ind(d1,d2, equal_var = False, nan_policy = 'omit')

if diffF2BMITtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus F2BMI levels shows a significant difference in the two cohorts.')
    EffectSize = cohend(d1,d2)
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')
else:
    print('There is no significant difference in F2BMI levels between the two cohorts.')
    
sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('F2 BMI  in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('F2 BMI')
plt.ylabel('Frequency')

#Now let us check for differences in bioimpedance between the cohorts

d1 = np.array(list(testingDataCL['F2bmpsc'].values))
d2 = np.array(list(testingDataBL['F2bmpsc'].values))
d1_desc  = testingDataCL['F2bmpsc'].describe()
d2_desc  = testingDataBL['F2bmpsc'].describe()
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

print(d1_desc, d2_desc)

print('The average F2 bioimpedance for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffF2bmpscTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffF2bmpscTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus F2bmpsc levels shows a significant difference in the two cohorts.')
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')
else:
    print('There is no significant difference in F2bmpsc levels between the two cohorts.')
    
sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('F2 bioimpedance in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('F2 bioimpedance')
plt.ylabel('Frequency')

#Now let's look at cholesterol indices, starting with hdl

d1 = np.array(list(testingDataCL['F2hdlch'].values))
d2 = np.array(list(testingDataBL['F2hdlch'].values))
d1_desc  = testingDataCL['F2hdlch'].describe()
d2_desc  = testingDataBL['F2hdlch'].describe()
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

print(d1_desc, d2_desc)

print('The average F2 hdlch for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus F2 hdl cholesterol levels shows a significant difference in the two cohorts.')
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')
else:
    print('There is no significant difference in F2 hdl cholesterol levels between the two cohorts.')
    
sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('F2 hdl cholesterolin CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('F2 hdl cholesterol')
plt.ylabel('Frequency')

d1 = np.array(list(testingDataCL['F2ldlch'].values))
d2 = np.array(list(testingDataBL['F2ldlch'].values))
d1_desc  = testingDataCL['F2ldlch'].describe()
d2_desc  = testingDataBL['F2ldlch'].describe()
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

print(d1_desc, d2_desc)

print('The average F2 ldlch for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus F2ldlch levels shows a significant difference in the two cohorts.')
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')
else:
    print('There is no significant difference in F2 ldl cholesterol levels between the two cohorts.')
    
sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('F2 ldl cholesterol in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('F2 ldl cholesterol')
plt.ylabel('Frequency')

d1 = np.array(list(testingDataCL['F2trig'].values))
d2 = np.array(list(testingDataBL['F2trig'].values))
d1 = np.log10(d1)
d2 = np.log10(d2)
d1_desc  = testingDataCL['F2trig'].describe()
d2_desc  = testingDataBL['F2trig'].describe()
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

print(d1_desc, d2_desc)

print('The average F2 trig for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus F2 triglcerides levels shows a significant difference in the two cohorts.')
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')
else:
    print('There is no significant difference in F2 triglycerides levels between the two cohorts.')
    
sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('F2 triglycerides in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('F2 triglycerides')
plt.ylabel('Frequency')

d1 = np.array(list(testingDataCL['F2gluc'].values))
d2 = np.array(list(testingDataBL['F2gluc'].values))
d1_desc  = testingDataCL['F2gluc'].describe()
d2_desc  = testingDataBL['F2gluc'].describe()
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

print(d1_desc, d2_desc)

print('The average F2 glucose for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus F2 glucose levels shows a significant difference in the two cohorts.')
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')
else:
    print('There is no significant difference in F2glucoselevels between the two cohorts.')
    
sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('F2 glucose in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('F2 glucose')
plt.ylabel('Frequency')

d1 = np.array(list(testingDataCL['F2crpu'].values))
d2 = np.array(list(testingDataBL['F2crpu'].values))
d1_desc  = testingDataCL['F2crpu'].describe()
d2_desc  = testingDataBL['F2crpu'].describe()
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

print(d1_desc, d2_desc)

print('The average F2 crpu for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus F2 crpu levels shows a significant difference in the two cohorts.')
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')
else:
    print('There is no significant difference in F2 crpu levels between the two cohorts.')
    
sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('F2 crpu in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('F2 crpu')
plt.ylabel('Frequency')

#Now let's look at differences in inflammation markers (interleukin, and tnfa). It says in one of the codebook
#that they should be logged but not clear why. Perhaps because the range is huge because of outliers. In that 
#case, we should log transform several other variables.
#For now, I am using raw values below.


d1 = np.array(list(testingDataCL['F2il6'].values))
d2 = np.array(list(testingDataBL['F2il6'].values))
d1 = np.log10(d1)
d2 = np.log10(d2)
d1_desc  = testingDataCL['F2il6'].describe()
d2_desc  = testingDataBL['F2il6'].describe()
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

print(d1_desc, d2_desc)

print('The average F2il6 for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus F2 interleukin 6 levels shows a significant difference in the two cohorts.')
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')
else:
    print('There is no significant difference in F2 interleukin 6 levels between the two cohorts.')
    
sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('F2 interleukin 6 in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('F2 interleukin 6')
plt.ylabel('Frequency')


d1 = np.array(list(testingDataCL['F2il1b'].values))
d2 = np.array(list(testingDataBL['F2il1b'].values))
d1 = np.log10(d1)
d2 = np.log10(d2)
d1_desc  = testingDataCL['F2il1b'].describe()
d2_desc  = testingDataBL['F2il1b'].describe()
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

print(d1_desc, d2_desc)

print('The average F2il1b for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus F2 interleukin 1b levels shows a significant difference in the two cohorts.')
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')
else:
    print('There is no significant difference in F2 interleukin 6 levels between the two cohorts.')
    
sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('F2 interleukin 1b in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('F2 interleukin 1b')
plt.ylabel('Frequency')

d1 = np.array(list(testingDataCL['F2tnfa'].values))
d2 = np.array(list(testingDataBL['F2tnfa'].values))
d1 = np.log10(np.array(list(testingDataCL['F2tnfa'].values)))
d2 = np.log10(np.array(list(testingDataBL['F2tnfa'].values)))
d1_desc  = testingDataCL['F2tnfa'].describe()
d2_desc  = testingDataBL['F2tnfa'].describe()
meanCL = np.nanmean(d1)
stdCL =np.nanstd(d1)
meanBL = np.nanmean(d2)
stdBL =np.nanstd(d2)

print(d1_desc, d2_desc)

print('The average F2tnfa for the CoLaus(non-brain) cohort is ' + repr(meanCL) + ' and in the BrainLaus cohort ' + repr(meanBL))

diffTtest =stats.ttest_ind(d1, d2, equal_var = False, nan_policy = 'omit')

if diffTtest[1] < 0.05:
    print('A two-sample t-test between non-BrainLaus and BrainLaus F2 tnfa levels shows a significant difference in the two cohorts.')
    if EffectSize > 0.2:
        print('The effect size is ' +repr (EffectSize) + ' which means there is a considerable difference between the two groups.')
    else:
        print('The effect size is too small to be meaningful.')
else:
    print('There is no significant difference in F2 tnfa levels between the two cohorts.')
    
sns.set(color_codes = True)
sns.distplot(d1[~np.isnan(d1)], label = 'CoLaus')
sns.distplot(d2[~np.isnan(d2)], label = 'BrainLaus')
plt.title('F2 tnfa in CoLaus and BrainLaus Participants')
plt.legend(loc ='upper right')
plt.xlabel('F2 tnfa')
plt.ylabel('Frequency')


testingData.to_csv('curatedCLData_20181108.csv')
testingDataBL.to_csv('DatatoMergeOnSES_20181108.csv')


#Last run on 08 11 2018 and man, plots is weird. Not weird on Jupyter. What a pain. Anyway, moving on. 
#Next stop merge on SES data 













