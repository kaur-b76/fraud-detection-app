# Fraud Detection Inspector - Detailed Project Explanation

## Introduction

Fraud Detection Inspector is a machine learning-based financial fraud analysis system developed using Python and Streamlit.

The purpose of this project is to simulate how banking systems and financial institutions analyze transactions to identify potentially fraudulent activity.

The application allows users to:
- browse transaction records,
- select any transaction from a large dataset,
- analyze the transaction using a trained machine learning model,
- compare predicted results with actual fraud labels,
- and visualize the factors influencing the prediction.

This project demonstrates:
- applied machine learning,
- financial transaction analysis,
- explainable AI,
- and interactive web application development.

---

# Problem Statement

Digital payment systems process millions of financial transactions every day.

Among these transactions:
- some are legitimate,
- while others are fraudulent.

Manual fraud detection is extremely difficult because:
- transaction volumes are massive,
- fraud patterns constantly evolve,
- and suspicious activities may appear similar to normal user behavior.

Machine learning helps solve this problem by learning patterns from historical transaction data and automatically identifying suspicious transactions.

This project builds a fraud detection system capable of:
- analyzing transaction patterns,
- detecting anomalies,
- and predicting whether a transaction is fraudulent.

---

# Type of Data Used

The project uses a financial transaction dataset.

Each row in the dataset represents:
> one individual money transaction.

The dataset contains transaction-related attributes such as:
- transaction type,
- transaction amount,
- sender balances,
- receiver balances,
- and fraud labels.

The data structure is similar to datasets used in:
- banking systems,
- mobile payment systems,
- digital wallets,
- and online transaction platforms.

---

# Understanding the Dataset

Each row describes the complete state of a financial transaction.

Example conceptually:

| Transaction | Sender | Receiver | Amount | Fraud Status |
|---|---|---|---|---|
| TXN001 | User A | User B | 5000 | Legitimate |
| TXN002 | Unknown | Unknown | 90000 | Fraud |

The machine learning model studies thousands or millions of such records to learn patterns associated with fraud.

---

# Meaning of Dataset Columns

## 1. `step`

Represents the unit of time for the transaction.

Typically:
- each step represents one hour.

Purpose:
- helps detect suspicious transaction timing,
- identifies rapid transaction bursts,
- and detects abnormal activity patterns.

Example:

| step |
|---|
| 1 |
| 2 |
| 3 |

---

## 2. `type`

Represents the type of transaction performed.

Possible values:
- CASH_IN
- CASH_OUT
- TRANSFER
- PAYMENT
- DEBIT

Example:

| type |
|---|
| TRANSFER |

Importance:
Certain fraud activities occur more frequently in specific transaction types such as:
- TRANSFER
- CASH_OUT

The model learns these behavioral patterns.

---

## 3. `amount`

Represents the amount of money transferred during the transaction.

Example:

| amount |
|---|
| 15000 |

Importance:
Large or unusual transaction amounts may indicate suspicious activity.

The model analyzes:
- transaction size,
- frequency,
- and abnormal transfer behavior.

---

## 4. `oldbalanceOrg`

Represents the sender's account balance before the transaction.

Example:

| oldbalanceOrg |
|---|
| 50000 |

Importance:
Used to verify whether transaction behavior appears normal.

---

## 5. `newbalanceOrig`

Represents the sender's account balance after the transaction.

Example:

| newbalanceOrig |
|---|
| 35000 |

Importance:
The model checks whether:
- balance deductions are logical,
- or whether abnormal balance changes occur.

---

## 6. `oldbalanceDest`

Represents the receiver's account balance before receiving money.

Example:

| oldbalanceDest |
|---|
| 2000 |

Importance:
Helps identify suspicious recipient account behavior.

---

## 7. `newbalanceDest`

Represents the receiver's account balance after receiving money.

Example:

| newbalanceDest |
|---|
| 7000 |

Importance:
Used to analyze:
- balance jumps,
- abnormal account growth,
- and suspicious transfer behavior.

---

## 8. `nameOrig`

Represents the sender account identifier.

Example:

| nameOrig |
|---|
| C123456 |

Importance:
Used to uniquely identify the source account.

During preprocessing:
- identifiers may be transformed,
- encoded,
- or removed depending on the ML pipeline.

---

## 9. `nameDest`

Represents the receiver account identifier.

Example:

| nameDest |
|---|
| M987654 |

Importance:
Used to identify destination accounts involved in transactions.

---

## 10. `isFraud`

Represents the target variable.

Values:
- `0` → legitimate transaction
- `1` → fraudulent transaction

Example:

| isFraud |
|---|
| 0 |
| 1 |

Importance:
This is the output label used during model training.

The machine learning model learns:
> which transaction patterns are associated with fraud.

---

# How the Machine Learning Model Works

The machine learning model does not think like a human investigator.

Instead, it learns statistical patterns from historical transaction data.

The model studies relationships between:
- transaction type,
- transfer amount,
- balance changes,
- account behavior,
- and fraud labels.

After training, the model predicts whether a new transaction is:
- fraudulent,
- or legitimate.

---

# Example of Model Learning

Suppose fraudulent transactions often have:
- very high transfer amounts,
- sender balances becoming nearly zero,
- transfer transaction types,
- abnormal balance changes.

The model gradually learns patterns such as:

```plaintext id="w0rq4v"
IF:
- transaction type = TRANSFER
- amount is unusually high
- sender balance becomes nearly empty

THEN:
transaction may be fraudulent
