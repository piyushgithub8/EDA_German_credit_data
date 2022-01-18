### Dataset Description:
The dataset is based on the “Statlog Dataset” from the UCI Machine Learning Repository. Columns of the dataset and their meaning are as follows;

Age (numeric)

Sex (text: male, female)

Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)

Housing (text: own, rent, or free)

Saving accounts (text - little, moderate, quite rich, rich)

Checking account (text - little, moderate, rich)

Credit amount (numeric, in Deutsche Mark)

Duration (numeric, in month)

Purpose (text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others

#### Skewness in Data

If the skewness is between -0.5 and 0.5, the data are fairly symmetrical If the skewness is between -1 and – 0.5 or between 0.5 and 1, the data are moderately skewed If the skewness is less than -1 or greater than 1, the data are highly skewed.


```python
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
gcd=pd.read_csv(r"C:\Users\Nikhil\Downloads\german_credit_data (2).csv") #reading dataset
```


```python
gcd.head() #getting peak of upper 5 observations
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Age</th>
      <th>Sex</th>
      <th>Job</th>
      <th>Housing</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>67</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>NaN</td>
      <td>little</td>
      <td>1169</td>
      <td>6</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>22</td>
      <td>female</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>moderate</td>
      <td>5951</td>
      <td>48</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>49</td>
      <td>male</td>
      <td>1</td>
      <td>own</td>
      <td>little</td>
      <td>NaN</td>
      <td>2096</td>
      <td>12</td>
      <td>education</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>45</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>7882</td>
      <td>42</td>
      <td>furniture/equipment</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>53</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>4870</td>
      <td>24</td>
      <td>car</td>
    </tr>
  </tbody>
</table>
</div>




```python
gcd.tail() #getting peak of lower 5 observations
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Age</th>
      <th>Sex</th>
      <th>Job</th>
      <th>Housing</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Purpose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>995</th>
      <td>995</td>
      <td>31</td>
      <td>female</td>
      <td>1</td>
      <td>own</td>
      <td>little</td>
      <td>NaN</td>
      <td>1736</td>
      <td>12</td>
      <td>furniture/equipment</td>
    </tr>
    <tr>
      <th>996</th>
      <td>996</td>
      <td>40</td>
      <td>male</td>
      <td>3</td>
      <td>own</td>
      <td>little</td>
      <td>little</td>
      <td>3857</td>
      <td>30</td>
      <td>car</td>
    </tr>
    <tr>
      <th>997</th>
      <td>997</td>
      <td>38</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>NaN</td>
      <td>804</td>
      <td>12</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>998</th>
      <td>998</td>
      <td>23</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>1845</td>
      <td>45</td>
      <td>radio/TV</td>
    </tr>
    <tr>
      <th>999</th>
      <td>999</td>
      <td>27</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>moderate</td>
      <td>moderate</td>
      <td>4576</td>
      <td>45</td>
      <td>car</td>
    </tr>
  </tbody>
</table>
</div>




```python
gcd.shape #getting shape of dataframe
```




    (1000, 10)



The dataset contains 1000 observations and 10 variables.


```python
gcd.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 10 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   Unnamed: 0        1000 non-null   int64 
     1   Age               1000 non-null   int64 
     2   Sex               1000 non-null   object
     3   Job               1000 non-null   int64 
     4   Housing           1000 non-null   object
     5   Saving accounts   817 non-null    object
     6   Checking account  606 non-null    object
     7   Credit amount     1000 non-null   int64 
     8   Duration          1000 non-null   int64 
     9   Purpose           1000 non-null   object
    dtypes: int64(5), object(5)
    memory usage: 58.7+ KB
    

The dataset contains 10 variables of which two has missing values. 
Saving accounts and Checking account contains missing values.


```python
gcd.isnull().sum() #checking number of null values
```




    Unnamed: 0            0
    Age                   0
    Sex                   0
    Job                   0
    Housing               0
    Saving accounts     183
    Checking account    394
    Credit amount         0
    Duration              0
    Purpose               0
    dtype: int64



It can be seen, Saving accounts has 183 null values and Checking account has 394 null values.

### Handling Missing Values


```python
gcd['Saving accounts'].values #to check the values of Saving accounts
```




    array([nan, 'little', 'little', 'little', 'little', nan, 'quite rich',
           'little', 'rich', 'little', 'little', 'little', 'little', 'little',
           'little', 'moderate', nan, nan, 'little', 'quite rich', 'little',
           'quite rich', 'little', 'moderate', nan, 'little', 'little',
           'rich', 'little', 'little', 'rich', 'little', 'moderate', nan,
           'little', 'little', 'little', 'little', 'little', 'little',
           'quite rich', 'quite rich', 'little', 'moderate', 'little',
           'little', 'quite rich', 'quite rich', 'little', 'moderate', nan,
           'little', 'little', nan, 'little', nan, nan, 'little', 'little',
           'little', 'little', nan, 'little', 'little', 'little', nan,
           'little', 'rich', 'little', nan, 'little', nan, 'little', 'little',
           nan, 'little', 'little', 'little', nan, 'little', nan,
           'quite rich', 'moderate', 'little', 'little', 'little', 'little',
           'moderate', 'moderate', 'little', 'little', 'little', nan, nan,
           'rich', 'little', nan, 'moderate', 'little', nan, 'moderate',
           'little', 'little', 'little', nan, 'little', 'little', 'little',
           nan, 'quite rich', 'moderate', 'little', 'little', 'little',
           'quite rich', nan, nan, nan, 'quite rich', 'quite rich', 'little',
           'little', 'little', 'little', nan, 'little', 'little', 'little',
           'little', 'little', nan, 'little', 'little', 'little', 'moderate',
           nan, 'rich', 'quite rich', nan, 'little', 'rich', 'little',
           'little', 'little', 'little', 'moderate', 'little', 'moderate',
           'little', 'rich', 'moderate', 'little', 'little', 'rich',
           'moderate', 'little', 'moderate', 'little', 'moderate', nan,
           'moderate', 'little', 'quite rich', 'little', 'quite rich',
           'quite rich', 'little', 'rich', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', nan, 'little',
           'quite rich', 'little', 'little', 'little', 'little', nan, 'rich',
           'little', 'little', 'little', 'little', 'moderate', 'little',
           'rich', 'moderate', 'little', 'little', 'moderate', 'little',
           'little', 'moderate', nan, 'little', 'quite rich', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'rich', nan, nan, 'little', 'little', nan, nan, 'little',
           'little', 'little', 'little', 'little', nan, 'little', nan,
           'little', 'little', 'rich', 'little', 'little', 'little', 'little',
           'quite rich', 'moderate', 'little', 'little', 'little', nan,
           'moderate', 'little', 'little', nan, 'little', 'little', 'little',
           'quite rich', 'little', 'little', 'moderate', 'little', 'little',
           'rich', 'little', 'little', 'moderate', nan, nan, 'little',
           'little', 'moderate', 'moderate', 'little', 'little', 'little',
           'little', 'little', 'little', nan, 'little', 'little', nan, nan,
           'quite rich', nan, 'little', 'little', 'little', 'little', nan,
           'little', 'moderate', 'rich', 'little', nan, nan, 'moderate',
           'little', 'little', 'moderate', 'little', 'little', 'little',
           'little', 'little', 'little', nan, 'little', nan, nan, 'little',
           'rich', 'little', 'little', nan, 'little', 'quite rich', 'rich',
           nan, 'moderate', 'little', 'little', nan, 'moderate', 'little',
           'little', nan, 'little', 'little', nan, 'little', 'little',
           'little', 'little', 'little', 'rich', 'little', 'little', nan,
           'rich', 'little', 'little', 'little', 'moderate', 'moderate',
           'moderate', 'little', 'little', 'little', nan, 'little', 'little',
           'little', 'little', 'quite rich', 'little', 'little', 'little',
           'little', 'quite rich', 'moderate', 'rich', 'little', 'little',
           nan, 'little', 'quite rich', 'little', nan, 'little', 'little',
           'little', nan, nan, 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', nan, 'little', 'little',
           nan, 'moderate', 'little', 'little', nan, 'little', 'moderate',
           nan, 'little', nan, 'little', 'moderate', 'little', nan, 'little',
           'quite rich', 'little', 'little', 'rich', 'little', 'little',
           'little', 'moderate', 'little', 'little', 'little', 'rich',
           'little', nan, 'little', 'little', nan, 'little', nan, 'little',
           'quite rich', 'quite rich', 'little', 'little', 'little',
           'quite rich', nan, 'little', 'little', nan, 'quite rich', nan,
           'rich', nan, 'little', 'moderate', nan, 'little', 'little', 'rich',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', nan, 'quite rich', 'rich', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', nan, 'little', 'little',
           nan, 'rich', nan, 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', nan, 'little', 'little', 'little', nan, 'little',
           'little', 'moderate', 'little', 'little', nan, 'little', 'little',
           'quite rich', 'little', 'moderate', 'little', 'little', 'little',
           'rich', 'rich', 'quite rich', 'little', 'little', 'little',
           'moderate', 'little', 'little', 'little', 'moderate', nan,
           'little', nan, 'moderate', 'little', 'little', 'moderate',
           'little', 'little', 'moderate', 'moderate', 'little', nan,
           'quite rich', 'moderate', 'little', 'moderate', 'little', 'little',
           'little', 'little', nan, 'little', 'little', 'little', 'moderate',
           nan, 'little', 'little', 'little', 'moderate', 'little', 'little',
           'moderate', 'little', 'little', 'little', 'little', 'moderate',
           'little', 'moderate', nan, 'little', nan, 'little', 'little',
           'little', 'little', 'little', nan, 'little', 'little', 'little',
           'little', 'little', 'little', nan, nan, 'quite rich', 'little',
           'moderate', 'little', 'little', 'moderate', nan, 'little',
           'little', nan, 'little', 'little', nan, nan, 'moderate', 'little',
           'rich', nan, 'little', 'little', 'little', nan, 'little', 'little',
           'little', 'little', nan, 'little', 'little', 'little', 'little',
           'little', 'little', nan, 'little', 'little', 'little', 'little',
           'little', 'little', nan, 'rich', 'little', nan, 'moderate',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'quite rich', 'little', 'moderate',
           'little', nan, 'moderate', 'moderate', 'rich', 'little', 'little',
           nan, nan, 'little', 'moderate', 'little', 'little', 'little', nan,
           'little', 'little', nan, 'little', 'moderate', 'quite rich', nan,
           'little', 'little', nan, 'little', 'little', 'little',
           'quite rich', 'little', 'little', 'little', 'little', 'little',
           'moderate', 'little', 'little', nan, 'little', 'quite rich',
           'little', 'little', 'little', 'little', 'little', 'moderate',
           'little', 'little', 'little', nan, 'little', 'little', 'little',
           nan, 'little', 'little', 'little', 'little', 'rich', 'little',
           'little', 'little', 'moderate', 'moderate', 'little', 'quite rich',
           'quite rich', 'little', 'little', 'moderate', 'little', 'little',
           'little', nan, 'little', 'little', 'moderate', nan, nan,
           'moderate', 'moderate', 'rich', 'little', 'moderate', 'moderate',
           nan, 'little', 'quite rich', 'little', 'little', 'little',
           'little', 'quite rich', 'little', 'moderate', 'moderate', 'little',
           'quite rich', 'moderate', 'little', 'little', 'little',
           'quite rich', 'little', nan, 'little', 'little', nan, nan,
           'little', nan, 'moderate', 'little', 'rich', 'little',
           'quite rich', 'little', 'rich', 'quite rich', 'little', 'little',
           'rich', 'little', 'little', 'little', 'rich', 'little', nan,
           'little', 'moderate', 'little', 'moderate', 'moderate', 'little',
           nan, 'quite rich', nan, 'little', 'little', 'little', nan,
           'little', 'little', 'little', 'moderate', 'little', 'rich', nan,
           'little', nan, 'little', 'little', nan, 'little', 'little', nan,
           'moderate', 'little', 'little', nan, 'little', 'little', nan,
           'little', 'little', 'moderate', 'quite rich', nan, 'little',
           'little', 'rich', 'little', 'little', 'rich', 'little', 'moderate',
           nan, 'rich', 'quite rich', nan, 'little', 'little', 'little', nan,
           nan, 'little', 'quite rich', 'moderate', nan, 'little', nan, nan,
           'little', 'little', 'little', nan, nan, 'little', 'little', nan,
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'quite rich', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', nan, nan, 'quite rich', 'little', 'little',
           nan, 'little', 'little', nan, 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', nan, nan,
           'little', 'little', 'little', 'little', nan, 'little', 'little',
           'little', 'moderate', nan, 'little', 'little', 'moderate', 'rich',
           'little', 'little', 'rich', 'little', 'quite rich', 'little',
           'little', nan, 'little', 'quite rich', 'little', 'moderate', nan,
           'little', 'rich', 'little', 'little', 'little', nan, 'little', nan,
           nan, 'little', 'little', 'little', nan, 'little', 'little', nan,
           'little', 'little', 'little', 'little', nan, nan, 'little', 'rich',
           nan, 'little', 'little', nan, nan, nan, 'little', 'little', nan,
           nan, 'little', nan, 'little', 'little', nan, 'rich', 'little',
           'little', 'moderate', 'little', 'moderate', 'little', 'little',
           'quite rich', 'little', 'little', 'little', 'little', 'little',
           'little', nan, 'little', 'little', 'little', 'little',
           'quite rich', 'little', 'moderate', 'little', 'little', 'little',
           nan, nan, 'little', nan, 'little', 'little', 'quite rich',
           'quite rich', nan, 'little', 'moderate', 'little', 'little',
           'quite rich', 'little', 'little', 'rich', nan, 'little', 'little',
           'moderate', 'little', 'little', nan, 'quite rich', 'little', nan,
           'quite rich', 'little', nan, 'little', 'moderate', 'little',
           'little', 'little', 'little', 'quite rich', 'little', nan,
           'little', 'moderate', 'little', 'little', 'moderate', 'little',
           'little', 'little', 'little', 'moderate', 'little', 'little', nan,
           'moderate', nan, 'little', nan, 'little', 'little', 'little',
           'little', 'moderate'], dtype=object)



It can be seen, the null values are present in the Saving accounts.


```python
#To check the distribution of values in Saving accounts
sns.countplot(x='Saving accounts',data=gcd)
plt.show()
```


    
![png](output_14_0.png)
    


The 'little' in Saving accounts has maximum number of distribution. Hence, it is possible to use mode of the Saving accounts and fill the null values.


```python
gcd['Saving accounts'].mode() #rechecking the mode of the Saving accounts variable
```




    0    little
    dtype: object




```python
gcd['Saving accounts'].fillna('little',inplace=True) #Filling null values with the mode of the Saving accounts
```


```python
gcd.info() #Checking the total values in the Saving accounts variable
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 10 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   Unnamed: 0        1000 non-null   int64 
     1   Age               1000 non-null   int64 
     2   Sex               1000 non-null   object
     3   Job               1000 non-null   int64 
     4   Housing           1000 non-null   object
     5   Saving accounts   1000 non-null   object
     6   Checking account  606 non-null    object
     7   Credit amount     1000 non-null   int64 
     8   Duration          1000 non-null   int64 
     9   Purpose           1000 non-null   object
    dtypes: int64(5), object(5)
    memory usage: 58.7+ KB
    

It can be seen that there are no more missing or null values in the 


```python
gcd['Saving accounts'].values #Rechecking if there are any missing or null values in Saving accounts
```




    array(['little', 'little', 'little', 'little', 'little', 'little',
           'quite rich', 'little', 'rich', 'little', 'little', 'little',
           'little', 'little', 'little', 'moderate', 'little', 'little',
           'little', 'quite rich', 'little', 'quite rich', 'little',
           'moderate', 'little', 'little', 'little', 'rich', 'little',
           'little', 'rich', 'little', 'moderate', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'quite rich',
           'quite rich', 'little', 'moderate', 'little', 'little',
           'quite rich', 'quite rich', 'little', 'moderate', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'rich', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'quite rich',
           'moderate', 'little', 'little', 'little', 'little', 'moderate',
           'moderate', 'little', 'little', 'little', 'little', 'little',
           'rich', 'little', 'little', 'moderate', 'little', 'little',
           'moderate', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'quite rich', 'moderate', 'little',
           'little', 'little', 'quite rich', 'little', 'little', 'little',
           'quite rich', 'quite rich', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'moderate', 'little',
           'rich', 'quite rich', 'little', 'little', 'rich', 'little',
           'little', 'little', 'little', 'moderate', 'little', 'moderate',
           'little', 'rich', 'moderate', 'little', 'little', 'rich',
           'moderate', 'little', 'moderate', 'little', 'moderate', 'little',
           'moderate', 'little', 'quite rich', 'little', 'quite rich',
           'quite rich', 'little', 'rich', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'quite rich', 'little', 'little', 'little', 'little', 'little',
           'rich', 'little', 'little', 'little', 'little', 'moderate',
           'little', 'rich', 'moderate', 'little', 'little', 'moderate',
           'little', 'little', 'moderate', 'little', 'little', 'quite rich',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'rich', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little', 'rich',
           'little', 'little', 'little', 'little', 'quite rich', 'moderate',
           'little', 'little', 'little', 'little', 'moderate', 'little',
           'little', 'little', 'little', 'little', 'little', 'quite rich',
           'little', 'little', 'moderate', 'little', 'little', 'rich',
           'little', 'little', 'moderate', 'little', 'little', 'little',
           'little', 'moderate', 'moderate', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'quite rich', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'moderate', 'rich',
           'little', 'little', 'little', 'moderate', 'little', 'little',
           'moderate', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little', 'rich',
           'little', 'little', 'little', 'little', 'quite rich', 'rich',
           'little', 'moderate', 'little', 'little', 'little', 'moderate',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'rich', 'little',
           'little', 'little', 'rich', 'little', 'little', 'little',
           'moderate', 'moderate', 'moderate', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'quite rich',
           'little', 'little', 'little', 'little', 'quite rich', 'moderate',
           'rich', 'little', 'little', 'little', 'little', 'quite rich',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'moderate', 'little', 'little', 'little', 'little',
           'moderate', 'little', 'little', 'little', 'little', 'moderate',
           'little', 'little', 'little', 'quite rich', 'little', 'little',
           'rich', 'little', 'little', 'little', 'moderate', 'little',
           'little', 'little', 'rich', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'quite rich', 'quite rich',
           'little', 'little', 'little', 'quite rich', 'little', 'little',
           'little', 'little', 'quite rich', 'little', 'rich', 'little',
           'little', 'moderate', 'little', 'little', 'little', 'rich',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'quite rich', 'rich', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'rich', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'moderate',
           'little', 'little', 'little', 'little', 'little', 'quite rich',
           'little', 'moderate', 'little', 'little', 'little', 'rich', 'rich',
           'quite rich', 'little', 'little', 'little', 'moderate', 'little',
           'little', 'little', 'moderate', 'little', 'little', 'little',
           'moderate', 'little', 'little', 'moderate', 'little', 'little',
           'moderate', 'moderate', 'little', 'little', 'quite rich',
           'moderate', 'little', 'moderate', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'moderate',
           'little', 'little', 'little', 'little', 'moderate', 'little',
           'little', 'moderate', 'little', 'little', 'little', 'little',
           'moderate', 'little', 'moderate', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'quite rich', 'little', 'moderate', 'little',
           'little', 'moderate', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'moderate', 'little',
           'rich', 'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little', 'rich',
           'little', 'little', 'moderate', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'quite rich', 'little', 'moderate', 'little', 'little', 'moderate',
           'moderate', 'rich', 'little', 'little', 'little', 'little',
           'little', 'moderate', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'moderate', 'quite rich',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'quite rich', 'little', 'little', 'little', 'little',
           'little', 'moderate', 'little', 'little', 'little', 'little',
           'quite rich', 'little', 'little', 'little', 'little', 'little',
           'moderate', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'rich', 'little', 'little', 'little', 'moderate',
           'moderate', 'little', 'quite rich', 'quite rich', 'little',
           'little', 'moderate', 'little', 'little', 'little', 'little',
           'little', 'little', 'moderate', 'little', 'little', 'moderate',
           'moderate', 'rich', 'little', 'moderate', 'moderate', 'little',
           'little', 'quite rich', 'little', 'little', 'little', 'little',
           'quite rich', 'little', 'moderate', 'moderate', 'little',
           'quite rich', 'moderate', 'little', 'little', 'little',
           'quite rich', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'moderate', 'little', 'rich',
           'little', 'quite rich', 'little', 'rich', 'quite rich', 'little',
           'little', 'rich', 'little', 'little', 'little', 'rich', 'little',
           'little', 'little', 'moderate', 'little', 'moderate', 'moderate',
           'little', 'little', 'quite rich', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'moderate',
           'little', 'rich', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'moderate', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'moderate', 'quite rich', 'little', 'little', 'little',
           'rich', 'little', 'little', 'rich', 'little', 'moderate', 'little',
           'rich', 'quite rich', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'quite rich', 'moderate', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'quite rich', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'quite rich', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'moderate', 'little',
           'little', 'little', 'moderate', 'rich', 'little', 'little', 'rich',
           'little', 'quite rich', 'little', 'little', 'little', 'little',
           'quite rich', 'little', 'moderate', 'little', 'little', 'rich',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'rich', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little', 'rich',
           'little', 'little', 'moderate', 'little', 'moderate', 'little',
           'little', 'quite rich', 'little', 'little', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'quite rich', 'little', 'moderate', 'little', 'little',
           'little', 'little', 'little', 'little', 'little', 'little',
           'little', 'quite rich', 'quite rich', 'little', 'little',
           'moderate', 'little', 'little', 'quite rich', 'little', 'little',
           'rich', 'little', 'little', 'little', 'moderate', 'little',
           'little', 'little', 'quite rich', 'little', 'little', 'quite rich',
           'little', 'little', 'little', 'moderate', 'little', 'little',
           'little', 'little', 'quite rich', 'little', 'little', 'little',
           'moderate', 'little', 'little', 'moderate', 'little', 'little',
           'little', 'little', 'moderate', 'little', 'little', 'little',
           'moderate', 'little', 'little', 'little', 'little', 'little',
           'little', 'little', 'moderate'], dtype=object)



The null values in the Saving accounts variable have been handled with the mode of the Saving accounts and shown.


```python
gcd['Checking account'].values #To check the values in the Current account
```




    array(['little', 'moderate', nan, 'little', 'little', nan, nan,
           'moderate', nan, 'moderate', 'moderate', 'little', 'moderate',
           'little', 'little', 'little', nan, 'little', 'moderate', nan, nan,
           'little', 'little', 'moderate', nan, 'little', nan, 'rich',
           'moderate', 'little', 'moderate', 'little', 'moderate', nan,
           'rich', 'moderate', nan, 'rich', 'rich', 'moderate', nan,
           'moderate', 'moderate', 'little', 'little', nan, nan, 'little',
           nan, nan, 'moderate', 'moderate', nan, nan, 'moderate', nan,
           'moderate', nan, 'rich', 'little', 'moderate', 'moderate',
           'moderate', 'moderate', nan, nan, nan, 'moderate', nan, nan, nan,
           nan, 'little', 'moderate', 'little', 'little', 'little',
           'moderate', nan, 'moderate', nan, nan, nan, 'little', 'little',
           nan, 'moderate', 'moderate', 'little', 'little', nan, 'little',
           nan, 'rich', 'moderate', 'moderate', nan, 'moderate', 'moderate',
           'moderate', nan, 'moderate', nan, 'moderate', nan, 'moderate', nan,
           'moderate', 'little', 'moderate', 'moderate', 'rich', 'moderate',
           nan, 'little', nan, 'little', 'little', 'little', 'moderate',
           'little', nan, nan, 'rich', 'moderate', 'little', 'little',
           'moderate', 'moderate', 'little', 'moderate', 'little', nan, nan,
           nan, nan, nan, 'moderate', 'moderate', 'rich', 'rich', 'moderate',
           'little', 'little', nan, 'moderate', 'little', nan, 'little', nan,
           nan, nan, 'rich', 'moderate', 'moderate', 'little', 'little',
           'little', 'moderate', nan, nan, nan, nan, 'moderate', nan, nan,
           'little', 'moderate', nan, 'moderate', 'little', nan, 'moderate',
           'moderate', 'little', nan, 'little', 'little', nan, 'little', nan,
           'moderate', 'little', nan, 'moderate', nan, 'moderate', 'moderate',
           'little', 'moderate', nan, 'moderate', 'moderate', nan, 'moderate',
           'moderate', nan, 'moderate', 'moderate', 'moderate', nan, 'little',
           nan, 'little', nan, 'little', nan, 'moderate', 'little', nan, nan,
           nan, 'little', 'rich', nan, 'moderate', 'little', 'rich', 'little',
           nan, 'moderate', 'little', nan, nan, nan, nan, 'moderate',
           'little', nan, 'little', 'rich', nan, nan, 'moderate', nan,
           'little', 'moderate', 'moderate', nan, 'little', 'little', nan,
           'little', nan, nan, nan, nan, nan, 'rich', nan, 'little', 'rich',
           'moderate', nan, 'moderate', 'moderate', nan, 'little', nan, nan,
           'little', 'little', 'little', nan, nan, 'moderate', nan, nan,
           'little', nan, nan, nan, 'moderate', 'moderate', 'little', nan,
           nan, 'little', nan, nan, nan, nan, 'rich', nan, 'moderate',
           'little', 'little', 'moderate', 'moderate', 'little', nan,
           'moderate', 'little', 'rich', nan, 'moderate', nan, nan, nan,
           'moderate', nan, 'moderate', 'rich', 'little', nan, nan, nan,
           'little', 'moderate', 'moderate', 'moderate', nan, 'rich',
           'moderate', 'rich', 'little', 'little', 'moderate', nan, 'little',
           'moderate', 'little', 'little', 'little', nan, 'little', nan, nan,
           'rich', 'moderate', 'little', nan, 'moderate', nan, 'little',
           'little', 'moderate', 'little', 'little', 'moderate', 'moderate',
           'little', 'moderate', 'moderate', 'rich', nan, 'moderate',
           'moderate', nan, 'moderate', nan, 'moderate', nan, 'little', nan,
           'moderate', nan, nan, nan, 'little', 'moderate', 'rich', 'rich',
           nan, 'little', nan, nan, 'little', 'little', 'moderate', nan, nan,
           nan, nan, 'moderate', 'little', nan, nan, 'moderate', nan,
           'little', 'moderate', nan, 'rich', nan, nan, nan, 'moderate',
           'moderate', nan, nan, 'moderate', 'little', 'little', nan,
           'moderate', 'little', 'little', 'moderate', nan, nan, 'moderate',
           nan, nan, 'moderate', 'moderate', nan, 'little', nan, 'rich',
           'moderate', nan, nan, nan, 'little', nan, 'little', 'little', nan,
           'moderate', nan, 'moderate', 'moderate', nan, 'moderate',
           'moderate', nan, nan, nan, 'little', nan, 'moderate', 'little',
           nan, 'little', 'moderate', nan, nan, 'little', 'rich', nan,
           'little', 'moderate', nan, 'moderate', nan, 'little', 'moderate',
           'rich', 'moderate', nan, nan, nan, nan, 'little', nan, 'little',
           'little', 'little', nan, 'little', 'little', 'moderate',
           'moderate', nan, 'little', 'little', nan, nan, nan, 'moderate',
           'little', 'little', nan, 'moderate', 'little', nan, 'rich',
           'moderate', 'little', 'moderate', 'moderate', 'little', nan, nan,
           'moderate', nan, nan, nan, nan, nan, 'moderate', nan, 'moderate',
           'little', 'little', 'moderate', nan, 'moderate', 'rich', 'little',
           'little', 'rich', 'moderate', 'little', nan, 'rich', 'moderate',
           nan, nan, 'little', nan, 'rich', 'moderate', nan, 'little',
           'little', nan, 'little', nan, nan, 'little', 'little', nan,
           'moderate', 'moderate', nan, nan, 'little', 'little', 'moderate',
           'moderate', nan, nan, nan, 'rich', 'little', 'moderate', 'little',
           'rich', 'moderate', nan, 'little', 'rich', nan, 'little', nan, nan,
           'little', nan, nan, nan, 'little', 'moderate', 'moderate',
           'moderate', 'moderate', nan, 'little', 'moderate', 'little',
           'little', 'rich', 'moderate', 'moderate', 'moderate', 'little',
           nan, 'moderate', 'little', 'little', nan, nan, 'little',
           'moderate', nan, 'moderate', nan, 'moderate', nan, 'moderate',
           'moderate', nan, 'moderate', nan, 'little', 'little', 'little',
           'little', 'little', 'little', 'moderate', nan, 'moderate',
           'little', 'moderate', 'little', 'moderate', nan, nan, 'moderate',
           'moderate', 'moderate', nan, 'rich', 'little', nan, 'moderate',
           nan, nan, 'little', 'rich', 'little', 'little', nan, 'moderate',
           'moderate', 'little', 'moderate', nan, 'moderate', nan, nan,
           'little', 'little', nan, 'rich', 'moderate', nan, nan, 'little',
           'little', 'moderate', nan, 'moderate', nan, nan, nan, nan,
           'little', 'little', 'moderate', nan, nan, 'little', nan, 'little',
           nan, 'rich', 'little', 'little', 'moderate', 'little', 'moderate',
           nan, 'little', 'moderate', nan, 'moderate', 'moderate', 'rich',
           'little', nan, 'moderate', 'rich', nan, 'moderate', nan, 'little',
           'rich', nan, nan, nan, nan, nan, nan, nan, 'moderate', 'little',
           nan, nan, nan, nan, nan, 'moderate', nan, 'rich', 'moderate', nan,
           'little', 'little', 'moderate', 'moderate', 'little', nan, nan,
           'moderate', nan, nan, 'rich', nan, 'little', 'rich', 'moderate',
           'moderate', nan, 'moderate', 'moderate', 'moderate', 'moderate',
           nan, 'little', nan, 'little', 'moderate', nan, nan, nan, 'rich',
           'moderate', 'rich', 'moderate', 'little', 'moderate', nan, nan,
           nan, 'little', 'moderate', 'rich', 'moderate', 'little',
           'moderate', nan, nan, 'moderate', 'moderate', 'little', nan,
           'moderate', 'little', 'moderate', nan, 'little', 'little',
           'little', 'little', 'little', nan, nan, 'little', 'little',
           'moderate', nan, nan, 'little', 'rich', 'rich', nan, 'little', nan,
           'little', 'little', nan, nan, 'moderate', 'little', nan,
           'moderate', nan, 'little', 'little', nan, nan, 'rich', 'little',
           nan, 'little', nan, 'moderate', 'moderate', nan, 'moderate',
           'moderate', 'moderate', 'moderate', nan, nan, 'moderate', 'little',
           'moderate', nan, nan, 'rich', nan, nan, 'little', nan, nan,
           'moderate', nan, 'moderate', 'little', nan, 'moderate', 'little',
           'moderate', nan, 'moderate', 'moderate', 'moderate', 'moderate',
           'little', 'little', 'little', 'moderate', nan, nan, 'little',
           'little', nan, 'rich', 'little', 'little', nan, 'little', 'little',
           nan, 'little', 'moderate', nan, 'little', 'little', 'moderate',
           'rich', 'little', nan, nan, 'little', nan, 'little', nan, nan, nan,
           nan, 'moderate', nan, nan, 'little', 'little', 'little', nan, nan,
           'little', nan, nan, nan, nan, 'little', nan, nan, nan, 'little',
           nan, nan, nan, 'little', nan, nan, 'little', nan, 'rich', 'little',
           nan, 'little', 'moderate', 'little', nan, 'little', nan, nan, nan,
           'moderate', nan, 'moderate', 'little', 'moderate', 'moderate', nan,
           nan, 'little', nan, 'little', 'moderate', nan, nan, 'little', nan,
           nan, 'little', 'little', nan, nan, nan, nan, 'little', 'little',
           'moderate', nan, 'moderate', nan, 'moderate', 'moderate', nan,
           'little', 'moderate', nan, 'little', 'little', 'little', nan, nan,
           'little', 'moderate', 'little', 'little', 'little', 'little', nan,
           'little', 'little', 'moderate', nan, nan, 'little', 'moderate',
           'rich', 'moderate', 'moderate', nan, nan, nan, nan, nan, 'little',
           'moderate', 'little', nan, nan, nan, 'moderate', 'little',
           'moderate', nan, 'little', 'little', 'rich', 'moderate', 'little',
           'moderate', nan, 'moderate', nan, nan, 'moderate', 'moderate',
           'moderate', nan, nan, 'little', 'moderate', nan, 'little',
           'little', nan, 'rich', 'moderate', 'moderate', nan, 'moderate',
           'moderate', nan, 'rich', 'little', nan, 'little', 'rich', nan,
           'little', 'moderate', nan, nan, 'little', 'little', nan, nan,
           'little', nan, 'little', 'moderate'], dtype=object)



It can be seen that there are null values present in the Current account variable.


```python
sns.countplot(x='Checking account', data=gcd) #To check the distribution of values in Current account
plt.show()
```


    
![png](output_24_0.png)
    


It can be seen that, the 'little' and 'moderate' has nearly same number of distribution. Filling null values with mode of the Checking account would not be possible.

Here, the Checking account, being the categorical data, has no contribution in the numerical relationship between other numerical variables hence it can be droped from the dataset. 


```python
gcd_drop=gcd.drop(['Checking account','Unnamed: 0'],axis=1) #dropping The Checking account from dataset and unnecessary variables
```


```python
gcd_drop.info() #Checing if the dataset has Checking account
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 8 columns):
     #   Column           Non-Null Count  Dtype 
    ---  ------           --------------  ----- 
     0   Age              1000 non-null   int64 
     1   Sex              1000 non-null   object
     2   Job              1000 non-null   int64 
     3   Housing          1000 non-null   object
     4   Saving accounts  1000 non-null   object
     5   Credit amount    1000 non-null   int64 
     6   Duration         1000 non-null   int64 
     7   Purpose          1000 non-null   object
    dtypes: int64(4), object(4)
    memory usage: 46.9+ KB
    

All the missing values have been handled.
There are no missing values in the dataset now.

Here the Job variable is categorical but it has been read as numeric. So it must be changed to object data type.


```python
gcd_drop['Job']=gcd_drop['Job'].map(str) #changing the data type of Job
```


```python
gcd_drop.dtypes #Rechecking the data type of the variables
```




    Age                 int64
    Sex                object
    Job                object
    Housing            object
    Saving accounts    object
    Credit amount       int64
    Duration            int64
    Purpose            object
    dtype: object



The data type of the Job variable has been corrected.

### Distribution of Numeric Variables


```python
gcd_drop.hist(figsize=(10,7)) #Checking distribution of all numeric variables
plt.show()
```


    
![png](output_33_0.png)
    


From the distribution of numeric variables it can be seen that, all the numeric variable distribution are right skewed.


```python
gcd_drop.skew() #To check the Skewness of the variable
```




    Age              1.020739
    Job             -0.374295
    Credit amount    1.949628
    Duration         1.094184
    dtype: float64



##### Skewness in Data

If the skewness is between -0.5 and 0.5, the data are fairly symmetrical If the skewness is between -1 and – 0.5 or between 0.5 and 1, the data are moderately skewed If the skewness is less than -1 or greater than 1, the data are highly skewed.

| Variable | Skewness |
|---|---|
| Age | Highly Right Skewed |
| Credit amount | Highly Right Skewed |
| Duration | Highly Right Skewed |

#### Handling Skewed Numeric Data


```python
sns.displot(np.log(gcd_drop['Age']),kind='hist',kde=True) #Using log function to handle skewed Age data
plt.show()
```


    
![png](output_39_0.png)
    



```python
np.log(gcd_drop['Age']).skew() #Skewness of variable after applying log function
```




    0.41625371561149377



The distribution of the Age variable is now fairly symmetrical.


```python
sns.displot(np.log(gcd_drop['Credit amount']),kind='hist',kde=True) #Using log function to handle skewed Credit amount data
plt.show()
```


    
![png](output_42_0.png)
    



```python
np.log(gcd_drop['Credit amount']).skew() #Skewness of variable after applying log function
```




    0.12928589230467



The distribution of the Credit amount variable is now fairly symmetrical.


```python
sns.displot(np.log(gcd_drop['Duration']),kind='hist',kde=True) #Using log function to handle skewed Duration data
plt.show()
```


    
![png](output_45_0.png)
    



```python
np.log(gcd_drop['Duration']).skew() #Skewness of variable after applying log function
```




    -0.1274144478919352



The distribution of the Duration variable is now fairly symmetrical.

##### Before and After Applying Log Function to Skewed Data

| Variable | Before Log Function | After Log Function |
|---|---|---|
| Age | Highly Right Skewed (1.02) | Fairly Symmetrical (0.41) |
| Credit amount | Highly Right Skewed (1.94) | Fairly Symmetrical (0.12) |
| Duration | Highly Right Skewed (1.09) | Fairly Symmetrical (-0.12) |

All the numeric highly skewed data have been handled.

### Outliers in Numeric Data


```python
sns.boxplot(gcd_drop['Age']) #Plotting boxplot to see if variable contains any outliers
plt.show()
```

    C:\Users\Nikhil\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_52_1.png)
    


The variable Age contains outliers.


```python
#Counting 5 point summary and handling the outliers in variable
Age_q1=np.quantile(gcd_drop['Age'],0.25)
Age_q2=np.quantile(gcd_drop['Age'],0.50)
Age_q3=np.quantile(gcd_drop['Age'],0.75)
Age_IQR=Age_q3 - Age_q1
Age_UpperWhisker= Age_q3+(Age_IQR*1.5)
Age_LowerWhisker= Age_q1-(Age_IQR*1.5)
```


```python
gcd_drop['Age']= np.where(gcd_drop['Age']>Age_UpperWhisker,
                          Age_UpperWhisker,gcd_drop['Age']) #Calculationg and Removing Outliers
```


```python
sns.boxplot(gcd_drop['Age']) #Plotting boxplot to see if there is still any outlier
plt.show()
```

    C:\Users\Nikhil\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_56_1.png)
    


It can be seen, the outliers in the variable Age have been handled.


```python
sns.boxplot(gcd_drop['Credit amount']) #Plotting boxplot to see if variable contains any outliers
plt.show()
```

    C:\Users\Nikhil\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_58_1.png)
    


The variable Age contains outliers.


```python
#Counting 5 point summary and handling the outliers in variable
CreditAmount_q1=np.quantile(gcd_drop['Credit amount'],0.25)
CreditAmount_q2=np.quantile(gcd_drop['Credit amount'],0.50)
CreditAmount_q3=np.quantile(gcd_drop['Credit amount'],0.75)
CreditAmount_IQR=CreditAmount_q3 - CreditAmount_q1
CreditAmount_UpperWhisker= CreditAmount_q3+(CreditAmount_IQR*1.5)
CreditAmount_LowerWhisker= CreditAmount_q1-(CreditAmount_IQR*1.5)
```


```python
gcd_drop['Credit amount']= np.where(gcd_drop['Credit amount']>CreditAmount_UpperWhisker, 
                          CreditAmount_UpperWhisker,gcd_drop['Credit amount']) #Calculationg and Removing Outliers
```


```python
sns.boxplot(gcd_drop['Credit amount']) #Plotting boxplot to see if there is still any outlier
plt.show()
```

    C:\Users\Nikhil\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_62_1.png)
    


It can be seen, the outliers in the variable Credit amount have been handled.


```python
sns.boxplot(gcd_drop['Duration']) #Plotting boxplot to see if variable contains any outliers
plt.show()
```

    C:\Users\Nikhil\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_64_1.png)
    


The variable Age contains outliers.


```python
#Counting 5 point summary and handling the outliers in variable
Duration_q1=np.quantile(gcd_drop['Duration'],0.25)
Duration_q2=np.quantile(gcd_drop['Duration'],0.50)
Duration_q3=np.quantile(gcd_drop['Duration'],0.75)
Duration_IQR=Duration_q3 - Duration_q1
Duration_UpperWhisker= Duration_q3+(Duration_IQR*1.5)
Duration_LowerWhisker= Duration_q1-(Duration_IQR*1.5)
```


```python
gcd_drop['Duration']= np.where(gcd_drop['Duration']>Duration_UpperWhisker,
                               Duration_UpperWhisker,gcd_drop['Duration']) #Calculationg and Removing Outliers
```


```python
sns.boxplot(gcd_drop['Duration']) #Plotting boxplot to see if there is still any outlier
plt.show()
```

    C:\Users\Nikhil\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_68_1.png)
    


It can be seen, the outliers in the variable Duration have been handled.

### Distribution Of All Variables

##### Distribution of Numeric Variables


```python
sns.displot(gcd['Age'],kind='hist',kde=True) #To Check distribution of variable
plt.show()
```


    
![png](output_72_0.png)
    


Here, the distribution of Age variable is shown.
It can be seen that, the age variable is highly distributed between age of 22 to 35, which results to right skewness.
This skewness of the Age variable has been handled previously by Log function.


```python
sns.displot(gcd['Credit amount'],kind='hist',kde=True,aspect=2) #To Check distribution of variable
plt.show()
```


    
![png](output_74_0.png)
    


Here, the distribution of Credit amount variable is shown.
It can be seen that, the Credit amount variable is highly distributed between amount of 1000 to 2500, which results to right skewness.
This skewness of the Credit amount variable has been handled previously by Log function.


```python
sns.displot(gcd['Duration'],kind='hist',kde=True) #To Check distribution of variable
plt.show()
```


    
![png](output_76_0.png)
    


Here, the distribution of Duration variable has shown.
It can be seen that, the Duration variable is varying highly between 12 to 22 months, which results to right skewness.
This skewness of the Duration variable has been handled previously by Log function.

##### Distribution of Categorical variables


```python
sns.countplot(gcd['Sex']) #To Check distribution of variable
plt.show()
```

    C:\Users\Nikhil\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_79_1.png)
    


The distribution of Sex variable is shown. The distribution of male is more than two times the distribution of female.


```python
sns.countplot(gcd['Job']) #To Check distribution of variable
plt.show()
```

    C:\Users\Nikhil\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_81_1.png)
    


The distribution of Job variable is shown. It can be seen the distribution of 'Skilled' people is high, followed by 
'Unskilled and Resident', 'Highly Skilled' and 'Unskilled and Non-Resident' people.


```python
sns.countplot(gcd['Housing']) #To Check distribution of variable
plt.show()
```

    C:\Users\Nikhil\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_83_1.png)
    


The distribution of Housing variable is shown. The distribution of people who own house are high followed by the people living on rent and the free housing.


```python
sns.countplot(gcd['Saving accounts']) #To Check distribution of variable
plt.show()
```

    C:\Users\Nikhil\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_85_1.png)
    


The distribution of Saving accounts variable is shown. The distribution of account with little savings are high followed by the moderate , quite rich and rich accounts.


```python
sns.countplot(gcd['Checking account']) #To Check distribution of variable
plt.show() 
```

    C:\Users\Nikhil\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_87_1.png)
    


The distribution of Checking account variable is shown. It can be seen that the distribution of little account is high followed by the moderate account and rich account.


```python
plt.figure(figsize=(12,5)) #To Check distribution of variable
sns.countplot(gcd['Purpose'])
plt.show()
```

    C:\Users\Nikhil\anaconda3\lib\site-packages\seaborn\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
      warnings.warn(
    


    
![png](output_89_1.png)
    


The distribution of Purpose variable is shown. From the distribution it is clear that, more number of people are saving for the expense of car, fowllowed by radio/TV, furniture/equipment, business and education. Very few people are saving for domestic appliances, repairs and vacation/others.

##### Numerical Vs. Categorical Plot


```python
sns.boxplot(x=gcd['Sex'], y=gcd['Age']) #To compare Age and Sex variable
plt.show()
```


    
![png](output_92_0.png)
    


The boxplot between Age and Sex has been plotted. It can be seen that there are more males with varying age with accounts compared to the number of females with accounts.


```python
sns.boxplot(x=gcd['Sex'], y=gcd['Credit amount']) #To compare Credit Amount and Sex variable
plt.show()
```


    
![png](output_94_0.png)
    


The boxplot between Credit amount and Sex has been plotted. It can be seen that maleshas more varying credit amount as compared to the females.


```python
sns.boxplot(x=gcd['Sex'], y=gcd['Duration']) #To compare Duration and Sex variable
plt.show()
```


    
![png](output_96_0.png)
    


The boxplot between Duration and Sex has been plotted. It can be seen that males has more variation in the duration of months than the females.

##### Numerical Vs. Numerical Plot


```python
sns.pairplot(gcd.drop(['Unnamed: 0','Job'],axis=1)) #Plotting Numeric against numeric variable
plt.show()
```


    
![png](output_99_0.png)
    


The numeric against numeric plot has been plotted.

#### Age Vs. Credit amount
It can be seen that, in Age Vs. Credit amount plot, more number of people in all age are having credit amount less than 5000. Few people with age between 20 to 50 are having credit amount between 5000 to 10000. Very few people between age 25 to 45 are having Credit amount between 10000 to 15000. 
#### Age Vs. Duration
It can be seen that, in Age Vs. Duration plot, most of the people of age between 18 to 55  are having duration of their account between 0 to 40 months. Few people of age more than 55, having duration of their account between 0 to 30 months. Very few people of age between 20 to 60, having duration of their account between 45 to 60 months.
#### Credit amount Vs. Duration
It can be seen that, in Credit amount Vs. Duration plot, most of the account with Credit amount under 10000 has a duration less than 30 months. Some account with Credit amount between 2000 to 15000 has a duration between 35 to 45 months. Very few accounts with Credit amount between 5000 to 15000 has a duration between 50 to 60 months.

### Conclusion:
Around 350 people between age 25 to 45, has credit amount around 10000 with duration less than 30 months.
