# AI Studio Challenge Project Title

> 💡 **Note for the team:** This is just a template. Update the above title with your AI Studio Challenge Project name. Remove all guidance notes and example text in this template and populate this README with your own content. You can work on this README throughout AI Studio, and get feedback from your AI Studio Coach and Challenge Advisor before finalizing it.  

---

### 👥 **Team Members**

**Example:**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Amina Waqar      | waqara1-ui    | Product Management, Data exploration, visualization, EDA                 |
| Jordan Ramirez   | @jramirez     | Data collection, exploratory data analysis (EDA), dataset documentation  |
| Amina Hassan     | @aminahassan  | Data preprocessing, feature engineering, data validation                 |
| Priya Mehta      | @pmehta       | Model selection, hyperparameter tuning, model training and optimization  |
| Chris Park       | @chrispark    | Model evaluation, performance analysis, results interpretation           |

---

## 🎯 **Project Highlights**

**Example:**

- Developed a machine learning model using `[model type/technique]` to address `[challenge project task]`.
- Achieved `[key metric or result]`, demonstrating `[value or impact]` for `[host company]`.
- Generated actionable insights to inform business decisions at `[host company or stakeholders]`.
- Implemented `[specific methodology]` to address industry constraints or expectations.

---

## 👩🏽‍💻 **Setup and Installation**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

---

## 🏗️ **Project Overview**

**Describe:**

- How this project is connected to the Break Through Tech AI Program
- Your AI Studio host company and the project objective and scope
- The real-world significance of the problem and the potential impact of your work

---

## 📊 **Data Exploration**

**You might consider describing the following (as applicable):**

* The dataset(s) used: origin, format, size, type of data
* Data exploration and preprocessing approaches
* Insights from your Exploratory Data Analysis (EDA)
* Challenges and assumptions when working with the dataset(s)

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)


---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.


** UPDATE 10/6/2026
# Wyndham Reservation & Loyalty Fraud Detection

[![Break Through Tech AI](https://img.shields.io/badge/Break%20Through%20Tech-AI%20Studio%202026-blue)](https://www.breakthroughtech.org/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey.svg)]()

## 1. Project Overview
This project is developed as part of the **Break Through Tech AI Studio / Wyndham Hotels & Resorts** collaboration. Our objective is to design a machine learning system and analyst-facing dashboard to detect and mitigate loyalty program abuse.

### Business Problem
* **The Exploit:** Bad actors book hotel reservations using invalid credit cards, intentionally trigger no-shows (`QNS`) or late cancellations after the 4:00 PM cutoff (`QXY`), harvest first-night loyalty point postings before hotels can zero out revenue, and rapidly redeem the points for Free Night Stays (`SRB`) to resell.
* **Operational Impact:** Fraudulent bookings block legitimate room inventory, cause revenue loss, and inflate loyalty program liabilities.
* **Current State:** Fraud teams rely on manual, post-incident Excel reviews.

### Core Deliverables
1. **Dual-Model ML Pipeline:**
   * **Model 1 (Booking-Time Risk):** Evaluates reservations at booking creation using advance lead time and account history to protect room inventory before arrival dates.
   * **Model 2 (Post-Stay Intervention):** Evaluates stays after qualification codes post to stop unauthorized point redemptions before free nights are consumed.
2. **Analyst Dashboard Prototype:** A 30-day investigation workspace that ranks risky transactions, displays explainable fraud indicators (SHAP-driven), and supports analyst workflow decisions.

---

## 2. Dataset & Framing

* **Prediction Target:** Binary classification at the transaction/reservation level (`is_fraud_event` ∈ {0, 1}), accompanied by continuous risk scores (0.0 to 1.0).
* **Data Scale (Revision 4):** 9.6 million combined reservation and stay records across May, June, and July (3.2M rows per month; 700,400 unique members with 400 labeled fraud accounts).
* **Class Imbalance:** Extreme imbalance (~4,800 fraud events out of 9.6M reservations).
* **Split Strategy:** Temporal month-to-month split (Train on May; Test/Validate on June/July) to prevent cross-month leakage from repeat fraud accounts.

---

## 3. Feature Engineering

Features are generated across three distinct behavioral pillars:
* **Timing & Lead Windows:** `booking_lead_days`, `stay_length_nights`, `single_night_ratio`, `same_day_booking_count`, `days_between_bookings`.
* **Qualification Codes:** Historical `qns_count`, `qxy_count`, `qns_qxy_rate`, `late_cancellation_rate` (exploiting the >4 PM cutoff).
* **Points & Redemption Velocity:** `points_earned`, `points_redeemed`, `qns_qxy_points_earned`, days from points posting to `SRB` redemption.

---

## 4. Modeling & Baseline Results

### Architecture
* **Baseline:** Scikit-Learn `LogisticRegression` pipeline with standard scaling for numeric features and one-hot encoding for categorical attributes.
* **Next Steps:** Gradient Boosting (`XGBoost`) with SHAP explainability.

### Preliminary Baseline Metrics (Model 2 - Held-Out Test Set)
* **Fraud Recall:** 100.0% (320 / 320 test fraud events caught)
* **Fraud Precision:** 97.3% (9 false positives across full test queue)
* **PR-AUC:** >99.9%
* **Operating Constraint:** Maintaining false positive rate (FPR) < 5% to protect genuine loyalty guests.

---

## 5. Dashboard Interface (Streamlit Prototype)

* **30-Day Activity Overview:** Daily volume of suspicious reservations and members.
* **Daily Drill-Down:** Filterable risk queue sorted by predicted risk score.
* **Audit Explanations:** Plain-English risk rationales (e.g., *1-night stay booked same-day with high historical QNS ratio*).
* **Analyst Feedback Mechanism:** Logged decisions (`Confirmed Fraud`, `False Positive`, `Pending`) for active learning retraining loops.

---

## 6. Repository Structure

```text
.
├── data/
│   ├── revision4/             # Raw CSV datasets (ignored by git)
│   └── *.parquet              # Processed feature stores (local only)
├── notebooks/
│   ├── combined_data.py       # Month consolidation
│   ├── data_cleaning.py       # Type casting & schema standardization
│   ├── booking_behavior.py    # Timing & lead window feature engineering
│   ├── qns_qxy_analysis.py    # Qualification code abuse analysis
│   └── points_redemption_analysis.py # Points earn vs. redemption velocity
├── dashboard/                 # Streamlit application
├── docs/                      # Data dictionaries & project briefs
├── requirements.txt           # Environment dependencies
└── README.md
