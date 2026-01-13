# 🧠 MLOps Pipeline for Fraud Detection using Amazon SageMaker

This project implements an **end-to-end Machine Learning workflow for fraud detection** using **AWS SageMaker**.  
The pipeline covers the complete ML lifecycle — from data preprocessing to model training, deployment, and monitoring — using managed AWS services.

> 🎯 Goal: Detect potentially fraudulent transactions using a cloud-hosted ML model.

---

## 🚀 Tech Stack

- **Amazon SageMaker Studio**
- **Amazon S3**
- **Python / Jupyter Notebook**
- **scikit-learn / XGBoost**
- **AWS IAM**
- **CloudWatch (optional monitoring)**

---

## 🔄 End-to-End Pipeline Flow

1️⃣ **Data Ingestion**
- Load dataset from local / S3 into SageMaker Studio

2️⃣ **Data Preprocessing**
- Clean data  
- Handle missing values  
- Encode categorical features  
- Split train/test  

3️⃣ **Model Training**
- Train ML model (e.g., XGBoost / Logistic Regression)
- Evaluate performance (Accuracy / AUC / Recall)

4️⃣ **Deployment**
- Deploy as a **real-time SageMaker endpoint**

5️⃣ **Prediction**
- Send sample input → receive fraud probability output

6️⃣ **(Optional) Monitoring**
- Track logs & metrics using CloudWatch

---

## 📁 Repository Structure
```
mlops-fraud-detection-sagemaker/
│
├── notebooks/ # Jupyter notebooks used in SageMaker
├── scripts/ # Python scripts (optional later)
├── images/ # Screenshots & architecture diagrams
└── README.md # Project documentation
```

## 🏗 Architecture (Conceptual)

User → SageMaker Studio Notebook
→ Data Stored in Amazon S3
→ Model Training in SageMaker
→ Model Deployed as Endpoint
→ Prediction Response Returned

---

## 📸 Project Setup & Execution Screenshots

### 1️⃣ Jupyter Space Creation
This screenshot shows the creation of a dedicated SageMaker Jupyter space named **MLOps-pipeline**, which is used to run the entire MLOps workflow.

![Jupyter Space Creation](images/01_create_jupyter_space_mlops_pipeline.png)

---

### 2️⃣ IAM Role Configuration
This screenshot confirms that the required IAM policies were successfully attached to the SageMaker execution role to enable access to AWS services.

![IAM Policies Attached](images/02_iam_policies_attached_success.png)

---

### 3️⃣ Notebook Execution in SageMaker Studio
This screenshot shows the Jupyter notebook running inside SageMaker Studio where the fraud detection MLOps pipeline code is executed.

![Notebook Execution](images/03_sagemaker_jupyter_notebook_run.png)


---

## 📊 Model — Fraud Detection

The dataset contains transaction-level records used to classify whether a transaction is **fraudulent or legitimate**.

Typical features include:
- transaction amount  
- device / user identifiers  
- transaction type  
- timestamp-based features  

Target:
fraud = 1
not fraud = 0

---

## 🧪 Evaluation Metrics

Key metrics monitored:

✔ Accuracy  
✔ Precision / Recall  
✔ ROC-AUC  

> ⚠ Fraud problems care heavily about **Recall / False Negatives**  
(because missing fraud is costly)

---

## 💰 Cost Awareness & Cleanup

To **avoid AWS billing**, all compute resources were shut down after use:

✔ Endpoints deleted  
✔ Notebook instances stopped  
✔ Training jobs completed  
✔ Only S3 storage retained  

---

## 📸 Project Evidence

Screenshots will be added here after execution:

- SageMaker Studio notebook  
- Training job  
- Model artifact  
- Endpoint deployed  
- Sample prediction output  

---

## 🎯 Learning Outcomes

✔ Hands-on experience with **cloud-based ML deployment**  
✔ Understanding of **MLOps workflow inside SageMaker**  
✔ Awareness of **resource & cost optimization**  
✔ Confidence working in real cloud environments  

---

## 🔮 Future Improvements

- Automate pipeline using **SageMaker Pipelines**
- Add **Model Registry**
- CI/CD using **CodePipeline / GitHub Actions**
- Add **Model Drift Monitoring**

---

## 👩‍💻 Author

**Kritika Aggarwal**

Feel free to connect 😊  
kritikaaggarwal2227@gmail.com
linkedin.com/in/kritika-aggarwal-734997249/

---

## ⭐ How to Use This Repo

This repository stores:

📂 Jupyter notebooks  
📂 Supporting scripts  
📂 Architecture diagram  
📂 Screenshots  
📄 Documentation  

Code is executed inside **Amazon SageMaker Studio** — NOT locally.

---
