# ETALON: Unsupervised Anomaly Detection and Interpretation for Multivariate Time Series
This is the implementation for the ETALON model architecture described in the paper: "ETALON: Unsupervised Anomaly Detection and Interpretation for Multivariate Time Series".

ETALON is a robust time series anomaly detection model which uses two discriminators to learn the normal pattern of multivariate time series and uses the reconstruction error to determine anomalies and interpret the root cause. ETALON has the following advantages: 1) Unsupervised; 2) Robustness: It achieve high performance in different real scenarios. 3) Effectiveness: It outperforms baselines in both accuracy and efficiency, achieving achieving an overall F1-score of 0.94, and very fast training (169 s per epoch) and inference (0.1 ms per entity); 4) Explainability: It pinpoints the dimensions caused the anomalies.

# Requirements

* Python 3.6
* Numpy (1.15.4)
* xlrd (1.2.0) # For data preprocess
* PyTorch (1.0.0)
* Scikit-learn (0.20.2)

# Dataset

* SMAP and MSL:

```
wget https://s3-us-west-2.amazonaws.com/telemanom/data.zip && unzip data.zip && rm data.zip

cd data && wget https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv
```

* SMD:

```
https://github.com/NetManAIOps/OmniAnomaly
```

* SWaT:

```
http://itrust.sutd.edu.sg/research/dataset
```

# Usage 

* Preprocess the data

```
python data_preprocess.py
```

* Run the code

```
python main.py <dataset>
```

where `<dataset>` is one of `SMAP`, `MSL`, `SMD`, `SWAT`.
