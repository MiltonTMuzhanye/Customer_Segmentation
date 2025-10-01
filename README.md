Customer Segmentation

Project Overview
This project groups customers into segments based on their shopping behavior. It helps businesses understand different customer types and create targeted marketing strategies for each group.

Business Problem
Businesses need to treat different customers differently. This project helps:
Identify high-value customers
Create personalized marketing campaigns
Improve customer retention
Allocate marketing budget effectively

Dataset
We use Online Retail data with:
541,909 transactions
4,339 unique customers
Features include:
Purchase dates and amounts
Product information
Customer IDs
Countries

Data Quality:
24.9% missing Customer IDs (removed)
2% cancellations (removed)

Final dataset: 397,924 transactions

What This Project Does
1 RFM Analysis
We analyze customers based on three key metrics:

Recency - Days since last purchase
Mean: 93 days
Range: 1 to 374 days

Frequency - Number of purchases
Mean: 4 purchases
Range: 1 to 210 purchases

Monetary - Total money spent
Mean: $2,054
Range: $0 to $280,206

2 Customer Segmentation
We create 8 RFM segments:

Segment	Customers	Description
Champions	947	Best customers - recent, frequent, high spending
Potential Loyalists	895	Good customers with growth potential
Loyal Customers	761	Regular, reliable customers
Can't Lose Them	757	High spenders but at risk of leaving
At Risk	466	Used to be good but becoming inactive
Promising	271	New customers showing potential
Others	242	Don't fit clear patterns
