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

# Import data
melbourneHousePrices = pd.read_csv("C:/Users/New/PycharmProjects/DataScience/Regression/Datasets/Melbourne_housing_FULL.csv")

# Summarise the data
print(melbourneHousePrices.head())
print('Number of rows:', melbourneHousePrices.shape[0])
melbourneHousePrices = melbourneHousePrices.dropna()
print('Number of rows remaining:', melbourneHousePrices.shape[0])
print(melbourneHousePrices.columns.values)
print(melbourneHousePrices.describe())

# We remove the NaNs and put both variables on a scatter plot
melbourneHouses = melbourneHousePrices[['BuildingArea', 'Price']]
sns.set_style("whitegrid")
sns.scatterplot(melbourneHouses.BuildingArea,melbourneHouses.Price)
plt.title('Price vs. Building Area')
plt.xlabel('Building Area ($m^2$)')
plt.ylabel('Price ($)')
plt.show()


print(melbourneHousePrices.loc[melbourneHousePrices['BuildingArea'].idxmax()])

# Let's drop this outlier and see how our scatter plot looks like
maxBuildingArea = melbourneHouses.BuildingArea.max()
melbourneHousesEdited = melbourneHouses[melbourneHouses['BuildingArea'] < maxBuildingArea]
sns.scatterplot(melbourneHousesEdited.BuildingArea, melbourneHousesEdited.Price)
plt.title('Price vs. Building Area')
plt.xlabel('Building Area ($m^2$)')
plt.ylabel('Price ($)')
plt.show()

# Let's investigate more outliers

largestBuildingAreas = melbourneHousePrices.nlargest(10, 'BuildingArea')
print(largestBuildingAreas[['Suburb','Distance','BuildingArea','Price']])

suburbCount = melbourneHousePrices['Suburb'].value_counts()
print(suburbCount[0:5])

# We can pick any of the top 5 counts of suburbs, for this example let's choose Richmond.
# We now analyse the plot a scatter plot of BuildingArea against Price for houses in Richmond.

richmondHouses = melbourneHousePrices[melbourneHousePrices['Suburb'] == 'Richmond']
sns.scatterplot(richmondHouses.BuildingArea, richmondHouses.Price)
plt.title('Richmond House Price vs. Building Area')
plt.xlabel('Building Area ($m^2$)')
plt.ylabel('Price ($)')
plt.show()

# There exists a positive correlation between Price and BuildingArea for
# houses in Richmond. This means we can start to build our regression model.









