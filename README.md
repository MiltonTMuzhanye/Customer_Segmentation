# Customer Segmentation

## Project Overview
This project groups customers into segments based on their shopping behavior. It helps businesses understand different customer types and create targeted marketing strategies for each group.

## Business Problem
Businesses need to treat different customers differently. This project helps:
* Identify high value customers
* Create personalized marketing campaigns
* Improve customer retention
* Allocate marketing budget effectively

## Dataset
We use Online Retail data with:
* 541,909 transactions
* 4,339 unique customers
* 
## Features include:
* Purchase dates and amounts
* Product information
* Customer IDs
* Countries

## Data Quality:
24.9% missing Customer IDs (removed)
2% cancellations (removed)

Final dataset: 397,924 transactions

## What This Project Does
**RFM Analysis**
We analyze customers based on three key metrics:

Recency - Days since last purchase
* Mean: 93 days
* Range: 1 to 374 days

Frequency - Number of purchases
* Mean: 4 purchases
* Range: 1 to 210 purchases

Monetary - Total money spent
* Mean: $2,054
* Range: $0 to $280,206

**Customer Segmentation**
We create 8 RFM segments using scoring:

| Segment	| Customers	| Description | 
|---------|-----------|-------------|
| Champions	| 947	| Best customers. Recent, frequent, high spending | 
| Potential Loyalists	| 895	| Good customers with growth potential | 
| Loyal Customers	| 761	| Regular, reliable customers | 
| Can't Lose Them	| 757	| High spenders but at risk of leaving |
| At Risk	| 466	| Used to be good but becoming inactive |
| Promising	| 271	| New customers showing potential |
| Others	| 242	| Don't fit clear patterns |



**Machine Learning Clustering**
Using K-means clustering, we identified 4 natural customer segments:

| Segment | Customers | Percentage | Key Characteristics |
|---------|-----------|------------|-------------------|
| VIP Customers | 558 | 12.9% | Recent (20 days), frequent (16 purchases), high spending ($10,007) |
| Regular Loyalists | 1,447 | 33.4% | Recent (45 days), moderate frequency (4 purchases), good spending ($1,667) |
| Occasional Shoppers | 1,392 | 32.1% | Moderate recency (58 days), low frequency (2 purchases), low spending ($393) |
| At Risk Customers | 941 | 21.7% | Very old recency (259 days), low frequency (1 purchase), low spending ($390) |


## Key results

### Model Performance
* Optimal Clusters: 4 (selected using elbow method and silhouette scores)
* Silhouette Score: 0.3805 (good cluster separation)
* Davies-Bouldin Index: 0.8570 (good cluster quality)

### Business Insights
1 VIP Customers (13%) (the most valuable customers who should receive exclusive offers and loyalty rewards)
2 Regular Loyalists (33%) (the core customer base, perfect for upselling and cross selling)
3 Occasional Shoppers (32%) (Have potential for growth with targeted promotions)
4 At Risk Customers (22%) (Need immediate win-back campaigns to prevent churn)  
