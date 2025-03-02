# Employee Segmentation Clustering Report

## **1. Clustering Goal**

The purpose of this analysis is to apply various clustering techniques to segment employees based on key characteristics. The clustering results aim to:

- Identify distinct employee groups with similar attributes.
- Uncover patterns related to employee retention risks.
- Provide HR teams with insights to improve employee satisfaction and reduce attrition.

The attrition variable is **not included** in the clustering process to ensure that the segmentation is based on inherent employee characteristics rather than forcing predefined labels.

## **2. Pre-processing**

### **Feature Selection (Removing Redundant Variables)**

To improve clustering effectiveness and reduce noise, certain variables were removed:

- **Dropped single-value columns**: `Over18`, `EmployeeCount`, `StandardHours`, `EmployeeNumber`.
- **Removed highly correlated features**: `JobLevel` was removed due to strong correlation with `Income`.

### **Handling Skewness**

Many numeric features were found to have high skewness, which can impact clustering performance. To address this:

- **Log transformation** was applied to reduce skewness and approximate a normal distribution.
- **Standardization (Z-score normalization)** was performed to center the data at 0 with a variance of 1.

### **One-Hot Encoding for Categorical Features**

Categorical variables were converted into numerical format using **one-hot encoding**, dropping the first category to avoid multicollinearity.

### **Principal Component Analysis (PCA)**

To improve computational efficiency and reduce dimensionality while retaining key information, PCA was applied. By analyzing the **PCA variance explanation plot**, we observed that PCA provided effective dimensionality reduction. We retained **90% of the variance**, resulting in **21 principal components** for further clustering analysis.

## **3. Clustering Techniques Applied**

### **K-Means Clustering**
- **Elbow Method** was applied to select the optimal `k` value. It showed that the clustering performed well when k=4 or 5. However, PCA scatter plots exhibited k=3 performed the best in terms of cluster boundary clarity, compactness and business interpretability. 
- The optimal number of clusters `k` was determined as considering both **Elbow Method** and **PCA Scatter Plots with different k values**.

### **Agglomerative Clustering**
- Multiple linkage criteria were tested (**ward, complete, single, average**).
- The **single linkage** and **average linkage** showed the highest silhouette scores while neither of them resulted in balanced clusters due to the chaining effect caused by their computational methods. The **ward linkage** and **complete linkage** should perform well on continuous HR dataset, but their silhouette scores are no better than KMeans clustering.

### **Other Models Considered and Discarded**
- **Gaussian Mixture Model (GMM)**: Despite its probabilistic nature, GMM struggled with clearly separating distinct employee segments, leading to overlapping clusters.
- **Spectral Clustering**: Computationally expensive with large datasets, and it did not significantly improve cluster quality.
- **DBSCAN**: Failed to produce meaningful clusters due to uneven density in the employee dataset.

The final clustering method selection was determined as **KMeans Clustering with 3 clusters** based on stability, interpretability, and ability to uncover meaningful employee segments.

## **4. Cluster Characteristics & Insights**

### **Cluster 0: Long-tenured Experienced Employees**
- This cluster contains **long-tenured** employees who **worked longest year in current role** with **lower salaries** and **slightly below average performance rating**.
- They have the **lowest attrition rates** which is likely due to **job security** or **familiarity**.
- While there is potential risks of **stagnation**, resulting in **future disengagement**.
- **Strategy:** Ensure continuous engagement, offering **career development programs** like **skill-building opportunities** to maintain retention.

### **Cluster 1: High-Risk Attrition Group**
- Employees in this cluster had **short tenure**, **low salaries**, and **limited career progression**.
- This group consisted mostly of **entry-level roles**, with fewer growth opportunities and dissatisfaction in their job roles.
- They showed **the highest attrition rate**, with an attrition rate of **20.35%**, which is likely due to early-stage job dissatisfaction or poor role fit.
- **Strategy:** Address salary concerns, improve job engagement, and provide **upskilling opportunities** to accelerate their developement or **job rotation opportunities** to help them find their best fit role to reduce attrition risk.

### **Cluster 2: High Performance & High Salary Employees**
- Employees in this group had **moderate tenure**, **the highest performance ratings**, and **the highest salaries**.
- They were high achievers, often promoted quickly, and highly valued by the company.
- Most employees in this cluster were in **high-responsibility roles**, including senior specialists and technical experts.
- **Strategy:** Go beyond financial incentives, focus more on **leadership development programs**, **retention bonuses**, providing challenging career opportunities and **structured career progression plans** to keep these high performers engaged and committed.

## **5. Conclusion**

The clustering analysis successfully identified meaningful employee segments. By leveraging different clustering techniques, we were able to:

- Gain a deeper understanding of employee distribution.
- Provide actionable insights for HR decision-making.
- Highlight retention risks and engagement opportunities.
