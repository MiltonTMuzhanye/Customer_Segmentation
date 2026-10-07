# Customer Segmentation System

A production-oriented customer segmentation system that transforms transactional retail data into actionable customer segments using **RFM analysis, behavioral feature engineering, K-Means clustering, batch inference, REST APIs, and an interactive Streamlit dashboard**.

The project is designed as an end-to-end machine learning system rather than a standalone clustering notebook.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Objectives](#objectives)
4. [Solution Architecture](#solution-architecture)
5. [End-to-End Workflow](#end-to-end-workflow)
6. [Dataset](#dataset)
7. [Data Quality](#data-quality)
8. [Data Preprocessing](#data-preprocessing)
9. [Feature Engineering](#feature-engineering)
10. [RFM Analysis](#rfm-analysis)
11. [Machine Learning Approach](#machine-learning-approach)
12. [Cluster Selection](#cluster-selection)
13. [Final Model](#final-model)
14. [Customer Segments](#customer-segments)
15. [Business Interpretation](#business-interpretation)
16. [Inference System](#inference-system)
17. [REST API](#rest-api)
18. [Streamlit Dashboard](#streamlit-dashboard)
19. [Project Structure](#project-structure)
20. [Generated Artifacts](#generated-artifacts)
21. [Testing](#testing)
22. [Installation](#installation)
23. [Running the Project](#running-the-project)
24. [Technology Stack](#technology-stack)
25. [Engineering Decisions](#engineering-decisions)
26. [Limitations](#limitations)
27. [Future Improvements](#future-improvements)
28. [Production Roadmap](#production-roadmap)
29. [Project Status](#project-status)

---

## Project Overview

This project analyzes customer purchasing behavior and automatically groups customers into meaningful behavioral segments.

The system starts with raw transactional data and progressively transforms it into customer-level behavioral features.

The resulting segmentation can support:

- Customer retention
- Churn prevention
- Customer lifetime value analysis
- Marketing campaign targeting
- Loyalty programs
- Cross-selling
- Upselling
- Win-back campaigns
- Customer prioritization
- Business intelligence

Unlike a basic clustering exercise, the project includes the surrounding engineering components required to turn a machine learning model into an application.

### Core Capabilities

- Transaction data ingestion
- Data validation
- Data cleaning
- RFM feature engineering
- Behavioral feature engineering
- Feature transformation
- Feature scaling
- K-Means clustering
- Multi-metric cluster evaluation
- Customer segment profiling
- Batch segmentation
- Model persistence
- REST API inference
- Streamlit dashboard
- Automated tests
- Configuration-driven processing

---

## Business Problem

Retail businesses typically have thousands of customers with very different purchasing behaviors.

Treating every customer identically makes it difficult to determine:

- Which customers are most valuable?
- Which customers are likely to churn?
- Which customers are highly engaged?
- Which customers require reactivation?
- Which customers deserve loyalty rewards?
- Which customers should receive targeted promotions?

The objective is therefore to discover natural groups of customers based on their purchasing behavior and translate those groups into actionable business strategies.

---

## Objectives

### Machine Learning Objectives

1. Convert transaction-level data into customer-level features.
2. Capture purchasing frequency, monetary value, recency, engagement and repeat behavior.
3. Identify meaningful behavioral groups without predefined labels.
4. Evaluate different numbers of clusters.
5. Select a practical clustering solution.
6. Persist the trained model and preprocessing artifacts.
7. Support inference on new customer records.

### Business Objectives

1. Identify high-value customers.
2. Identify customers showing signs of inactivity.
3. Identify customers requiring retention campaigns.
4. Identify new or one-time customers.
5. Provide actionable recommendations for each segment.

---

## Solution Architecture

```text
                         RAW TRANSACTION DATA
                                  |
                                  v
                         DATA INGESTION
                                  |
                                  v
                         DATA VALIDATION
                                  |
                                  v
                       DATA PREPROCESSING
                                  |
                                  v
                     CUSTOMER-LEVEL FEATURES
                                  |
                  +---------------+---------------+
                  |                               |
                  v                               v
             RFM FEATURES                   BEHAVIORAL FEATURES
                  |                               |
                  +---------------+---------------+
                                  |
                                  v
                       FEATURE ENGINEERING
                                  |
                                  v
                      FEATURE TRANSFORMATION
                                  |
                                  v
                         FEATURE SCALING
                                  |
                                  v
                       K-MEANS CLUSTERING
                                  |
                                  v
                       CLUSTER EVALUATION
                                  |
                                  v
                       SEGMENT PROFILING
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
              BATCH INFERENCE              MODEL ARTIFACTS
                    |                           |
                    v                           v
             CUSTOMER SEGMENTS          SAVED MODEL + SCALER
                    |
              +-----+------+
              |            |
              v            v
          REST API     STREAMLIT APP
---

## Data Preprocessing

The preprocessing pipeline converts the raw transaction dataset into a clean dataset suitable for customer-level analysis.

### Processing Steps

1. Load the raw Excel dataset.
2. Validate the expected schema.
3. Remove transactions without `CustomerID`.
4. Remove cancellation transactions.
5. Remove quantities below the configured minimum.
6. Remove transactions with non-positive unit prices.
7. Calculate transaction-level monetary value.
8. Save the cleaned transaction dataset.

### Transaction Value

For every valid transaction:

```text
Amount = Quantity × UnitPrice
---

## Data Preprocessing

The preprocessing pipeline converts the raw transaction dataset into a clean dataset suitable for customer-level analysis.

### Processing Steps

1. Load the raw Excel dataset.
2. Validate the expected schema.
3. Remove transactions without `CustomerID`.
4. Remove cancellation transactions.
5. Remove quantities below the configured minimum.
6. Remove transactions with non-positive unit prices.
7. Calculate transaction-level monetary value.
8. Save the cleaned transaction dataset.

### Transaction Value

For every valid transaction:

```text
Amount = Quantity × UnitPrice
---

## Data Preprocessing

The preprocessing pipeline converts the raw transaction dataset into a clean dataset suitable for customer-level analysis.

### Processing Steps

1. Load the raw Excel dataset.
2. Validate the expected schema.
3. Remove transactions without `CustomerID`.
4. Remove cancellation transactions.
5. Remove quantities below the configured minimum.
6. Remove transactions with non-positive unit prices.
7. Calculate transaction-level monetary value.
8. Save the cleaned transaction dataset.

### Transaction Value

For every valid transaction:

```text
Amount = Quantity × UnitPrice


---

## Data Preprocessing

The preprocessing pipeline converts the raw transaction dataset into a clean dataset suitable for customer-level analysis.

### Processing Steps

1. Load the raw Excel dataset.
2. Validate the expected schema.
3. Remove transactions without `CustomerID`.
4. Remove cancellation transactions.
5. Remove quantities below the configured minimum.
6. Remove transactions with non-positive unit prices.
7. Calculate transaction-level monetary value.
8. Save the cleaned transaction dataset.

### Transaction Value

For every valid transaction:

```text
Amount = Quantity × UnitPrice



## Customer Segments

The final K-Means model produces five operational customer segments.

| Cluster | Segment | Customers | Share | Primary Strategy |
|---:|---|---:|---:|---|
| 0 | Dormant / Lost | 718 | 16.55% | Win-back and reactivation |
| 1 | New / One-Time | 807 | 18.60% | Second-purchase conversion |
| 2 | At Risk / Needs Attention | 875 | 20.17% | Retention and targeted incentives |
| 3 | Champions / VIP | 1,197 | 27.59% | VIP rewards and premium treatment |
| 4 | Loyal Customers | 741 | 17.08% | Loyalty, cross-sell and upsell |

Total:

```text
4,338 customers