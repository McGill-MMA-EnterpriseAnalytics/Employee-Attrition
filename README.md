# Employee-Attrition

### Introduction

Imagine losing a valuable employee and facing not just their salary, but up to **200%** of that amount in hidden costs—from recruiting and onboarding to lost productivity and diminished team morale. Studies by the Society for Human Resource Management confirm that replacing an employee can cost **50–200%** of their annual salary, while Gallup reveals nearly **51%** of U.S. employees are actively seeking new opportunities. For a median-sized S&P 500 company, McKinsey estimates the **annual cost of employee disengagement and attrition at \$228 million**.

Yet, turnover is preventable. Surveys show that more than **half** of voluntary exits could be avoided if employers proactively addressed issues like career development, compensation, or management quality. To support these proactive strategies, our project unites **predictive modeling**, **clustering analysis**, and **causal inference** to not only forecast attrition risk but also pinpoint and validate the genuine drivers behind employees’ decisions to leave. By combining data-driven insights with robust causal methods, HR leaders can better understand **who** is most at risk, **why** they are likely to leave, and **what** interventions can most effectively retain them—ultimately curbing these staggering turnover costs.


### 🔍 Objectives**

**Predictive Modeling** pinpoints which employees are most at risk of leaving, enabling HR to concentrate retention efforts where they matter most. By integrating comprehensive preprocessing, feature engineering, and advanced model tuning, it provides an early-warning mechanism that highlights subtle signs of disengagement or job dissatisfaction before employees make the decision to exit.

**Causal Inference** dives deeper into **why** employees leave, separating genuine drivers from mere correlations. It tests factors like overtime, income level, or promotion frequency through rigorous statistical frameworks, giving leadership confidence that certain interventions (e.g., reducing overtime, boosting job involvement) will truly lower attrition. This data-driven clarity helps ensure limited resources are allocated to the highest-impact initiatives.

**Clustering** segments the workforce into naturally occurring groups based on shared attributes—without using attrition status as a clue. As a result, it uncovers hidden subpopulations (e.g., early-career strivers or high-performing veterans) that require customized approaches. These insights let HR craft tailored policies, benefits, and career pathways that resonate with each segment’s unique needs.

 ### Methodology

#### Exploratory Data Analysis (EDA)

The project started with data familiarization, confirming dataset completeness, checking for missing values or duplicates, and examining basic statistics. Subsequently, univariate and bivariate analyses were conducted to visualize distributions (e.g., histograms and boxplots) and reveal outliers, skewness, and potential relationships (e.g., MonthlyIncome, WorkLifeBalance) tied to attrition. A correlation heatmap highlighted key attributes influencing turnover, guiding subsequent decisions on feature selection. Lastly, preliminary clustering tests (e.g., elbow method, hierarchical dendrogram) offered insights into latent employee groupings, laying the groundwork for advanced segmentation and deeper modeling.

Feature Distributions – Salary, total working years, job satisfaction trends.

Attrition Breakdown – Department, gender, and promotion impact.

Correlation Analysis – Identifying key drivers of attrition.

Clustering Readiness – PCA and optimal K-means cluster selection.

Causal Inference Preparation – Treatment vs. control group comparisons.

Further in-depth data transformations and final dataset preparation take place in the Preprocessing notebook within the predictive modeling workflow


#### Predictive Classification Modeling for Employee Attrition

These notebooks build, tune, and evaluate end-to-end machine learning pipelines to predict employee attrition. Our goal is to identify the best pipeline configuration by experimenting with various preprocessing methods and classifiers. Key steps include:

- **Manual Preprocessing & Feature Engineering:**  
  Cleaning the data, handling missing values, and creating new features and transforming existing ones to enhance predictive power. This part consists of a manual exploration of the data to make informed assumptions on the appropriate preprocessing and feature selection steps, before they are put to the test in the pipeline contruction and tuning part of the project.

- **Pipeline Construction & Tuning:**  
  Evaluating different strategies (e.g., numeric transformations, scaling, categorical encoding, outlier removal, and feature selection) and fine-tuning classifiers like Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, and MLP using GridSearchCV and RandomizedSearchCV.

- **Model Evaluation & Ensemble Learning:**  
  Assessing performance with metrics such as F2 score, accuracy, recall, and ROC-AUC, analyzing learning curves, and combining top models into a stacking ensemble for improved generalization.

- **Business Insights:**  
  Interpreting key model coefficients to uncover factors driving attrition, thereby informing proactive retention strategies.

This streamlined approach not only aims to optimize predictive performance but also provides actionable insights for effective workforce management.

#### Causal Inference Analysis for Employee Attrition

This notebook uses the DoWhy framework to uncover the true causal drivers behind employee attrition. Instead of merely identifying correlations, our analysis pinpoints factors that genuinely influence the likelihood of an employee leaving.

- **Modeling:**  
  We construct causal graphs for each potential driver—including overtime, job involvement, work-life balance, time since promotion, income level, relationship satisfaction, and past job mobility—to visualize the assumed causal relationships.

- **Identification & Estimation:**  
  Using backdoor adjustment methods, we identify the causal effects and then estimate them through linear regression, treating each factor as a binary variable (e.g., overtime: Yes=1, No=0).

- **Refutation:**  
  To validate our findings, we perform robustness checks (e.g., random common cause testing and data subset analysis) ensuring that our causal estimates are reliable and not due to unobserved confounding.

##### Business Insights

The analysis reveals that:
- **Overtime** has a strong causal impact on increasing attrition.
- **High job involvement** and **good work-life balance** significantly reduce the likelihood of leaving.
- **Past job mobility** is a predictor of future attrition.

These insights can help HR teams design targeted interventions—such as managing overtime, boosting employee engagement, and improving work-life balance—to effectively reduce turnover.

This notebook provides a clear, actionable framework to move beyond correlation and drive strategic, data-backed decisions for employee retention.


### Employee Segmentation via Clustering

This part of the project focuses on applying clustering techniques to segment employees based on their inherent characteristics. The goal is to identify distinct groups without biasing the process by including attrition as a feature, thereby uncovering natural patterns in the workforce. The methodology comprises the following key steps:

- **Data Preprocessing:**  
  - Remove duplicates and drop redundant features.
  - Address skewness through log transformation followed by standardization.
  - Encode categorical variables using One-Hot Encoding.

- **Dimensionality Reduction:**  
  - Apply PCA to capture 90% of the variance, reducing noise and facilitating more robust clustering.

- **Clustering Evaluation:**  
  - Experiment with various algorithms (KMeans, Agglomerative Clustering, DBSCAN, Spectral Clustering, and Mean Shift).
  - Use silhouette scores and the elbow method to evaluate cluster quality and determine the optimal number of clusters.
  - Visualize clusters in 2D and 3D using PCA-reduced data.

- **Cluster Profiling:**  
  - Assess cluster characteristics and explore differences in external variables (e.g., attrition rates) to inform potential HR strategies.

This systematic approach ensures that employee segmentation is driven by data-driven insights, laying the groundwork for targeted retention initiatives.


### Environment Setup

To ensure reproducibility and consistency, please follow these steps to build the project environment using the provided requirements file:

1. **Install Python:**  
   Ensure you have Python 3.x installed on your system.

2. **Create a Virtual Environment:**  
   - **Windows:**  
     ```bash
     python -m venv env
     ```
   - **macOS/Linux:**  
     ```bash
     python3 -m venv env
     ```

3. **Activate the Virtual Environment:**  
   - **Windows:**  
     ```bash
     env\Scripts\activate
     ```
   - **macOS/Linux:**  
     ```bash
     source env/bin/activate
     ```

4. **Install Dependencies:**  
   With the virtual environment activated, install all required packages using:
   ```bash
   pip install -r requirements.txt
   ```

5. **Deactivate the Environment (when finished):**  
   Simply run:
   ```bash
   deactivate
   ```

Following these steps will set up your environment with the necessary libraries for the project.

### Dataset Source

This project uses the IBM HR Analytics Attrition dataset, which is available on Kaggle. You can download the dataset from the following link:

[IBM HR Analytics Attrition Dataset on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
