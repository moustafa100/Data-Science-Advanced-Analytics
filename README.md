# Data-Science-Advanced-Analytics

**PowerCo** is a major European gas and electricity utility serving corporate, SME (Small & Medium Enterprise), and residential customers. After energy market liberalization in Europe, PowerCo experienced significant SME churn. This project, with BCG, focuses on diagnosing churn and developing actionable retention strategies.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Solution Approach](#solution-approach)
- [Repository Structure](#repository-structure)
- [Data Description](#data-description)
- [Getting Started](#getting-started)
- [How to Use](#how-to-use)
- [Results & Insights](#results--insights)
- [License](#license)

---

## Project Overview

The repository provides an end-to-end analytics pipeline for **SME customer churn diagnosis**, including data cleaning, feature engineering, exploratory analysis, machine learning modeling, and business impact evaluation.

---

## Business Problem

European energy market liberalization has caused unprecedented SME customer churn for PowerCo. The challenge: understanding churn drivers and shaping evidence-based retention actions.

---

## Solution Approach

- **EDA and Data Cleaning:** Scrutinize, cleanse, and prepare the data.
- **Feature Engineering:** Create new, meaningful, predictive customer features.
- **Modeling:** Apply advanced analytics and ML to anticipate churn risk.
- **Business Impact Analysis:** Estimate the financial implications of retention and discount strategies.

---

## Repository Structure

| Name                             | Description                                 |
|----------------------------------|---------------------------------------------|
| `EDA and Data cleaning.py`       | Core script for exploration and cleaning    |
| `Feature engineering.py`         | Script for constructing predictive features |
| `Business Impact of discount.py` | Business impact and discount analytics      |
| `ml_case_training_data.csv`      | Main structured dataset                     |
| `ml_case_training_hist_data.csv` | Historical actions and customer statuses    |
| `ml_case_training_output.csv`    | Outcome labels or model outputs             |
| `history_data.pkl`               | Preprocessed history data (pickled)         |
| `model_data.pkl`                 | Feature-ready modeling data (pickled)       |
| `LICENSE`                        | Project license (MIT)                       |
| `README.md`                      | Project documentation (this file)           |

---

## Data Description

- **ml_case_training_data.csv:** Core customer and contract data
- **ml_case_training_hist_data.csv:** Event history (metering, contracts, etc.)
- **ml_case_training_output.csv:** Churn labels or validation outcomes

Key variables: customer ID, dates, tariffs, energy consumption, region, event histories, churn status.

---

## Getting Started

1. **Clone the repository**


