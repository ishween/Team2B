# Salesforce 2B CRM Intelligence Assistant

---

### 👥 **Team Members**

| Name               | GitHub Handle    | Contribution   |  
|--------------------|------------------|----------------|  
| Nissi Otoo         | @nssim516        |  Data Cleaning, Semantic Search Implementation, Account Summary Agent & Email Drafting Agent, GBDT Model Implementation, Project Lead |
| Khin Yuupar Myat   | @hera-myat       | Data Cleaning, LLM Integration, Gradio Interface Development, CatBoost Model Implementation, CRM Dashboard Visualizations, Task Assignment & Team Coordination |  
| Jaren Taznim       | @jren55          | Data Standardization, LLM Integration, Account Summary Agent, Random Forest Model Testing, Gradio Interface |  
| Zainab Ahmed       | @zainabahmed4    | Lead Scoring Model, GBDT Model Selection & Optimization, Advanced Analytics Engine, CRM Dashboard Visualizations |  
| Jean-Parnell Louis | @jean-parnellone | Feature Engineering, Account Health Scoring, Multi-Modal Data Indexing, Export & Reporting System, GitHub Documentation |
| Kayla Cheng        | @klhrcn          | Exploratory Data Analysis, Semantic Search Implementation, LLM Integration, Office Hours Coordination, Advanced Analytics Engine |
| Fanizza T. Tahir   | @axzhir           | X             |

---

## 🎯 **Project Highlights**
This project is part of the **Break Through Tech AI Program** in partnership with **Salesforce** as our AI Studio host company. Below are some of our project highlights:

- Developed a machine learning model using Gradient Boosting Decision Trees (GBDT) and Random Forest classifiers to address lead scoring, opportunity win prediction, and account health assessment.
- Achieved **87% accuracy in lead conversion prediction and 85% in deal closure forecasting** for Salesforce sales and marketing teams.
- Generated actionable insights including pipeline performance analysis, **at-risk account identification**, conversion rate trends, and **revenue forecasting** to inform business decisions at Salesforce and enable data-driven sales strategies.
- Implemented **semantic search with ChromaDB vector embeddings** and **three specialized Gen AI agents** using advanced prompt engineering to address enterprise-scale Customer Resource Management (CRM) data retrieval challenges and **automate time-intensive sales analysis tasks**.


---

## 👩🏽‍💻 **Setup and Installation**

### Must Haves
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/ishween/Team2B.git
cd Team2B
```

2. **Create and activate a virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Set up environment variables**
Create a `.env` file in the root directory:
```
LLM_API_KEY=your_api_key_here
```

5. **Access the dataset**
- Download the CRM Sales Opportunities dataset from [Kaggle](https://www.kaggle.com/datasets/innocentmfa/crm-sales-opportunities)
- Place the CSV file in the `data/` directory

6. **Run the application**
```bash
# Run the Gradio interface
python app.py

# Or run individual components
python agents/account_summary_agent.py
python agents/email_drafting_agent.py
python agents/insight_generation_agent.py
```


---

## 🏗️ **Project Overview**

### 💡 Project Objective
Develop an AI assistant that helps sales and marketing teams better understand their customer relationships by analyzing CRM data. The system supports critical business functions including:
- Automating lead scoring and conversion prediction
- Predicting opportunity outcomes and deal closure success
- Assessing account health and identifying at-risk customers
- Generating contextual, personalized sales emails
- Providing intelligent business insights through natural language queries

### 🗂️ Significance
Customer Relationship Management is the backbone of modern sales operations. However, sales teams often struggle with:
- **Information overload**: Sifting through massive amounts of customer data
- **Time constraints**: Manually analyzing accounts and drafting personalized communications
- **Missed opportunities**: Failing to identify at-risk accounts or high-value prospects
- **Inconsistent insights**: Lack of standardized analysis across the sales organization

Our solution addresses these challenges by providing an AI-powered analysis of CRM data through a conversational interface so that sales and marketing teams can make quick and effective decisions.

### ⚙️ Technical Architecture
The system has three main components:

1. **Machine Learning Pipeline**: Predictive models for lead scoring, opportunity forecasting, and account health assessment
2. **Gen AI Agents**: Three specialized agents powered by advanced prompt engineering
   - Account Summary Agent: Comprehensive account analysis with risk assessment
   - Email Drafting Agent: Context-aware email generation with multiple tones and types
   - Insight Generation Agent: Pipeline analysis and strategic recommendations
3. **Gradio Chat Interface**: Natural language query processing with semantic search and multi-agent routing

---

## 📊 **Data Exploration**

### Dataset Overview
- **Source**: [Kaggle - CRM Sales Opportunities Dataset](https://www.kaggle.com/datasets/innocentmfa/crm-sales-opportunities)
- **Format**: CSV

### Data Preprocessing Approach
1. **Data Cleaning** (Completed Sept 3, 2025)
   - Addressed missing values using appropriate imputation strategies
   - Identified and handled outliers using IQR method
   - Renamed columns for consistency and clarity
   - Removed duplicate records
   - Fixed typos and data entry errors

2. **Data Standardization** (Completed Sept 12, 2025)
   - Standardized categorical variables with consistent naming conventions
   - Normalized numerical features using StandardScaler
   - Encoded categorical features for machine learning models

3. **Feature Engineering** (Completed Sept 19, 2025)
   - Created derived features such as deal velocity, engagement scores
   - Calculated time-based features (days in pipeline, time since last activity)
   - Developed composite metrics for account health scoring


### Data Structure
The dataset contains:
- **Customer Demographics**: Company information, industry sectors, geographic locations
- **Sales Pipeline Data**: Opportunity stages, deal amounts, close dates, probability scores
- **Engagement Metrics**: Interaction logs, activity history, marketing campaign responses
- **Account Information**: Revenue figures, account health indicators, relationship history

**Potential visualizations to include: [NEED TO ADD]**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## 🧠 **Model Development**

We built three machine learning models that significantly improved sales intelligence outcomes:

- **Lead Scoring Model**: Boosted prediction accuracy from a **42% baseline to 87%**, using a Random Forest model.
- **Opportunity Win Prediction**: Delivered **85% accuracy** for forecasting deal closures, enabling early identification of at-risk opportunities.
- **Account Health Scoring**: Created a regression-based health score (0–100) with an **R^2 of 0.82**, incorporating revenue, engagement, and support metrics.

### Semantic Search & GenAI

- Implemented **semantic search with ChromaDB + SentenceTransformer embeddings** for natural-language queries.
- Integrated LLMs using **advanced prompt engineering** with dynamic context injection for reliability and consistency.
- Built **three specialized GenAI agents**:
  - **Account Summary Agent** for real-time insights and risk assessment  
  - **Email Drafting Agent** for personalized, context-aware sales outreach  
  - **Insight Generation Agent** for pipeline trends, at-risk deal detection, and forecast analysis  

### Optimization & Training

- Applied **feature engineering, Recursive Feature Elimination (RFE), and GridSearchCV** to optimize model performance.
- Used 80/20 splits and cross-validation, consistently achieving **85%+ accuracy** across key predictive models.

---

## 📈 **Results & Key Findings**

| Model | Metric | Score | Impact |
|-------|--------|-------|--------|
| Lead Scoring | Accuracy | 87% | Improved lead prioritization |
| Lead Scoring | F1-Score | 0.85 | Balanced precision-recall |
| Opportunity Prediction | Accuracy | 85% | Early risk detection |
| Opportunity Prediction | Precision | 0.83 | Reduced false positives |
| Account Health | R^2 | 0.82 | Reliable health assessment |
| Account Health | RMSE | 12.5 | Acceptable error margin |

### 💼 Business Impact

**Sales Efficiency Improvements**:
- Reduced time spent on manual account analysis
- Automated email drafting saves nearly 1 hour per sales rep per day
- Instant access to pipeline insights enables faster decision-making

**Revenue Optimization**:
- Early identification of at-risk accounts enables proactive intervention
- Top opportunity identification helps sales teams prioritize high-value deals
- Improved lead scoring increases conversion rates by focusing on qualified prospects

**User Satisfaction**:
- Gradio interface for ease of accessibility and user interface (UI)
- Natural language query processing eliminates need for complex CRM navigation
- Real-time response system provides instant insights


**Potential visualizations to include [NEED TO ADD]:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

### Model Limitations
- Models are trained on past data, so performance can fade without regular retraining.
- Some sales signals (like relationship strength or competitor moves) are hard to capture numerically.
- LLM-generated insights are helpful but still need human oversight for important communications.

### Future Enhancements
- Enable real-time Salesforce integration with continuous updates and automated alerts.
- Improve personalization with multilingual options and a potential voice interface.

---

## 📝 **License [NEED TO ADD]**

If applicable, indicate how your project can be used by others by specifying and linking to an open source license type (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type.

**Example:**  
This project is licensed under the MIT License.

---

## 📄 **References**

1. Kaggle CRM Sales Opportunities Dataset - https://www.kaggle.com/datasets/innocentmfa/crm-sales-opportunities
2. Sentence Transformers Documentation - https://www.sbert.net/
3. Gradio Documentation - https://www.gradio.app/docs/
4. ChromaDB Documentation - https://docs.trychroma.com/docs/overview/introduction

---

## 🙏 Acknowledgements

We’re really grateful to the people who made this project possible:

- **Ishween Kaur**, Senior Software Engineer at Salesforce — your technical knowledge (and honesty!) helped us make smarter decisions and stay grounded. You met with us every other week, asked the questions we didn’t even know to ask, and somehow managed to challenge us without ever making us feel small. Thank you for caring enough to push us and for giving us permission to take up space confidently!

- **Leah Dsouza**, our AI Studio Coach — you kept us organized, encouraged, and sane. Your check-ins and support really helped us move forward when things felt overwhelming and you always made time for us, even when we dropped in with last-minute questions. You helped us see the bigger picture and kept us grounded throughout.

- **Salesforce** — thank you for trusting us with a meaningful problem and giving us access to the kind of data, context, and challenges that you can’t get from a classroom.

- The **Break Through Tech community** — we’re grateful for a space that didn’t just hand us resources, but actually taught us how to navigate ambiguity, handle messy data, and build something end-to-end.
