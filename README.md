# Fraud Detection with Apache Spark

## Nigeria University of Technology and Management (NUTM)

**Big Data Technologies (NUTDTS 805) | MSc Data Science | 2025/2026**

End-to-end distributed Big Data pipeline for real-time financial 
fraud detection. Built with HDFS and Apache Spark (MLlib + PySpark) 
using the PaySim Synthetic Financial Dataset to detect fraudulent 
mobile money transactions and identify the top high-risk behavioural 
indicators.

---

## Team Members

| Name | Student ID | GitHub | Responsibility |
|------|-----------|--------|----------------|
| Ibukunoluwa Adeleke | 252325001 | @Ibukun-ola | Data Ingestion & Feature Engineering |
| Chioma Agu | 252325002 | @Aguchioma | Model Training |
| Bolanle Olaleye | 252325007 | @bolz-coder | Evaluation |
| All Group Members | ... | ... | Reporting |

**Lecturer:** Dr. Isah Charles Saidu

---

## Tech Stack

- Apache Hadoop 3.4.2 (HDFS)
- Apache Spark 3.5.0 (MLlib + PySpark)
- Python 3.12.3
- Java OpenJDK 11
- Dataset: PaySim Synthetic Financial Dataset (Kaggle)

---

## Repository Structure
src/
├── ingestion.py              # Load PaySim into HDFS, schema validation, EDA
├── feature_engineering.py    # Spark Windowing, velocity features, data cleaning
├── model.py                  # Random Forest pipeline, class imbalance handling
└── evaluation.py             # Recall score, confusion matrix, top 3 features
data/                         # Dataset download instructions
report/                       # Final report and findings
notebooks/                    # Supporting notebooks

---

## Project Overview

A mobile money provider needs to identify fraudulent transactions 
in real-time. The dataset contains 6.3 million transactions with 
a 1:100 fraud-to-legitimate ratio, requiring sophisticated sampling 
techniques and a high Recall classifier to minimise missed fraud.

### Key Findings
- Total Transactions: 6,362,620
- Fraud Ratio: 0.1291%
- Primary Metric: Recall Score
- Top 3 fraud indicators: (to be updated after evaluation)
