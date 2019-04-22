import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Configuring print options
desired_width=320
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)

melbourneHousePrices = pd.read_csv("C:/Users/New/PycharmProjects/DataScience/Regression/Datasets/Melbourne_housing_FULL.csv")

print(melbourneHousePrices.columns)

# We will be selecting houses that belong in Richmond.
richmondHouses = melbourneHousePrices[melbourneHousePrices['Suburb'] == 'Richmond'].dropna()

# Boxplot to examine outliers
sns.set_style("whitegrid")
sns.boxplot(richmondHouses.BuildingArea)
plt.xlabel('Building Area ($m^2$)')
plt.title('Building Area of Richmond Houses')
plt.show()

# There exists a few outliers, let's check them out, we do this by using the IQR
print(richmondHouses.loc[richmondHouses.BuildingArea.idxmin()])

buildingArea = richmondHouses.BuildingArea
q1, q3 = buildingArea.quantile(0.25), buildingArea.quantile(0.75)
print(q1, q3)
IQR = q3 - q1
print(IQR)

outliers = richmondHouses[(buildingArea < q1 - IQR*1.5) | (buildingArea > q3 + IQR*1.5)]
print(outliers[['Rooms','Price','BuildingArea','Address','Distance']].sort_values('BuildingArea',ascending=False))

richmondHouses = richmondHouses[richmondHouses.BuildingArea > buildingArea.min()]
sns.boxplot(richmondHouses.BuildingArea)
plt.xlabel('Building Area ($m^2$)')
plt.title('Building Area of Richmond Houses')
plt.show()

# Compute the linear regression model
target = richmondHouses.Price
predictor = richmondHouses.BuildingArea
predictors = sm.add_constant(predictor)
model = sm.OLS(target,predictors).fit()
print(model.summary())

sns.regplot(predictor,target,line_kws={'color':'g'},ci=None)
plt.title('Richmond House Price vs. Building Area')
plt.xlabel('Building Area ($m^2$)')
plt.ylabel('Price ($)')
plt.show()

standardResiduals = pd.Series(model.resid_pearson, name='Standardised Residuals')
sns.regplot(predictor,standardResiduals, fit_reg=False)
plt.title('Residuals')
plt.xlabel('Building Area ($m^2$)')
plt.ylabel('Standardised Residuals')
plt.show()
