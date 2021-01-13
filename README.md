# ETALON
## ETALON: Unsupervised Anomaly Detection and Interpretation for Multivariate Time Series
This is the implementation for the ETALON model architecture described in the paper: "ETALON: Unsupervised Anomaly Detection and Interpretation for Multivariate Time Series".

ETALON is a robust time series anomaly detection model which uses two discriminators to learn the normal pattern of multivariate time series and uses the reconstruction error to determine anomalies and interpret the root cause. ETALON has the following advantages: 1) Unsupervised; 2) Robustness: It achieve high performance in different real scenarios. 3) Effectiveness: It outperforms baselines in both accuracy and efficiency, achieving achieving an overall F1-score of 0.94, and very fast training (169 s per epoch) and inference (0.1 ms per entity); 4) Explainability: It pinpoints the dimensions caused the anomalies.
## Getting Started
