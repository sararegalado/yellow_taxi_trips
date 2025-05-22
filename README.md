# 🗽 NYC Yellow Taxi Trips - Big Data Analytics with PySpark

## 📌 Project Overview

This project analyzes the **New York City Yellow Taxi Trips** dataset using the Spark Dataframe, MLlib and Spark Structure Streaming APIs, combining batch analytics, machine learning, and real-time data streaming to derive insights and build predictive models. The project was completed as part of an academic Big Data course (Tratamiento de Grandes Volúmenes de Datos), following strict guidelines regarding distributed computing, analytical depth, and infrastructure deployment on AWS.

---

## 📁 Repository Structure

```
yellow_taxi_trips/
│
├── MLLib/src/                              # ML models using Spark MLlib                   
├── analytical_queries/                     # Analytical queries using Spark                                                   Dataframe API
		├── output     
├── data/                        
├── data_preprocessing/          
├── spark_structured_streaming/             # Streaming jobs using Spark
																							Structured Streaming and Kafka
├── .gitignore
└── .DS_Store
```

---

## 🚀 Goals & Scope

The main objectives of this project were to:

- ✅ Deploy a **Hadoop cluster** on AWS Academy with HDFS and YARN
- ✅ Execute **analytical queries** on the NYC Taxi dataset using Spark DataFrame API
- ✅ Develop and evaluate **machine learning models** using Spark MLlib
- ✅ Implement **streaming jobs** using Spark Structured Streaming and Kafka

---

## 📊 Dataset & Motivation

We selected the **NYC Yellow Taxi Trips** dataset due to its:

- Rich set of features (trip distance, pickup/drop-off times, payment type, etc.)
- Volume suitable for Big Data processing
- Relevance for urban mobility, pricing, and behavior analytics

Our analytical goals include understanding traffic patterns, fare distributions, trip durations, and geographical behaviors.

---

## 🔧 Infrastructure Setup

The project infrastructure was deployed on AWS Academy using the following architecture:

- 🔹 1 Master Node (NameNode, ResourceManager)
- 🔹 5 Worker Nodes (DataNodes, NodeManagers)
- 🔹 **Spark** configured to run in YARN cluster mode
- 🔹 Event generator node (Kafka + Docker) to simulate real-time streaming events

---

## 📈 Analytical Queries

Five analytical queries were implemented, focusing on domain-specific insights and using a variety of aggregations:

1. Revenue per hour by zones
2. Most frequent pairs of pickup and drop-off locations
3. Average speed of the trip
4. Average tip by hour
5. Number of trips by day of the week and by hour

All queries were executed on the Hadoop cluster and results stored in HDFS.

---

## 🤖 Machine Learning

ML tasks focused on predicting **tip amount** and **trip duration** using regression models. Implemented algorithms include:

- Generalized Regression
- Decision Tree Regression
- Random Forest Regression
- GBT Regressor

### Evaluation Metrics:

- RMSE
- R² Score
- MAE

Data preprocessing included handling missing values, feature engineering, and normalization.

---

## 🌐 Structured Streaming with Kafka

Real-time streaming queries were implemented with Spark Structured Streaming, Kafka, and Docker.

### Streaming Tasks:

1. **Top 10 Countries Accessing Finance Articles (Last 15 min)**
2. **Real-Time Device Usage Trends**
3. **Active Sessions per Country (Last 5 min)**
4. **Breaking News Spike Detection** (Articles >250 views in 10 mins)
5. **Bot-like Behavior Detection** (Users >20 views/min)

---

## 📎 How to Run

```
spark-submit --archives /home/ec2-user/pyspark_conda_env.tar.gz#environment --master yarn taxi_trips/analytics.py 
```

```bash
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 --master yarn [your-query.py](http://your-query.py/)
```
