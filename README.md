# Financial Health Assessment Tool

This application helps you assess the financial health of a company based on key financial ratios. 
By inputting financial metrics like ROE, ROA, and debt ratios, the tool will classify the financial 
condition as Good, Average, or Poor.

## How to Use
1. Enter the financial ratios in the input fields
2. Click "Submit" to get the prediction
3. Review the results and probability scores

## Key Features
- Predicts financial health classification (Good, Average, Poor)
- Shows probability scores for each category
- Provides basic interpretation of results

## Notes
- ROE and ROA should be entered as decimals (e.g., 5% = 0.05)
- For best results, use data from established companies with complete financial statements
- This is a simplified model and should be used as one of many tools in financial analysis

## Model Information
The model is a RandomForest classifier trained on financial data from publicly traded companies.
It evaluates financial health based on patterns learned from historical data.
