README – Energy Consumption Forecasting with Pattern Discovery Overview This project analyzes smart home energy consumption data to identify usage patterns and forecast future energy demand. The pipeline combines unsupervised learning (clustering) with supervised machine learning models to improve prediction accuracy. Dataset The dataset Energy Consumption Dataset for Smart Homes contains timestamp-based records of appliance electricity usage including: Ventilador PC AC Lampara TV Each row represents energy usage at a specific time. Workflow

Data Loading The dataset is loaded using Pandas from an Excel file.
Data Preprocessing Timestamp is converted into datetime format. New time-based features are extracted: Hour Day Month Total energy consumption is calculated by summing appliance usage.
Exploratory Analysis A correlation matrix is generated using Seaborn to understand relationships between variables.
Feature Extraction Energy consumption patterns are derived by grouping data by hour and calculating: Average consumption Variance Peak consumption
Clustering K-Means clustering is applied to group hours with similar consumption behavior. Clusters help identify different energy usage patterns.
Data Preparation for Prediction Features used for prediction: Appliance usage Hour of the day Data is split into training (80%) and testing (20%) sets.
Machine Learning Models Two supervised regression models are trained: Random Forest Regressor Gradient Boosting Regressor These models predict total energy consumption.
Model Evaluation Model performance is evaluated using: RMSE (Root Mean Squared Error) MAE (Mean Absolute Error) R² Score
Feature Importance F-Score analysis identifies statistically important predictors. Random Forest feature importance shows which appliances contribute most to energy consumption.
Visualization The pipeline generates: Correlation heatmap Cluster visualization Feature importance graph Actual vs predicted energy consumption plot Outcome The project helps utility providers: Identify energy consumption patterns Segment usage behaviors using clustering Predict future energy demand using machine learning models.
