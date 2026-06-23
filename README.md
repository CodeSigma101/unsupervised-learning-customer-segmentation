# Unsupervised Learning: Customer Segmentation Hub

An end-to-end data science pipeline and responsive analytical dashboard that discovers hidden mathematical groupings in retail data using unsupervised machine learning.

## Project Architecture

1. **Feature Engineering**: Aggregates unstructured transaction logs into high-density customer profile records with 20+ performance attributes.
2. **Dimensionality Reduction**: Utilizes Principal Component Analysis (PCA) to compress high-dimensional feature spaces into 3 primary spatial vectors.
3. **Mathematical Clustering**: Applies K-Means clustering, verified through Elbow Method and Silhouette Score optimizations, to partition the audience into 4 separate operational marketing personas.
4. **Interactive BI UI Dashboard**: Built with Streamlit to provide visual segmentation coordinates, target profile metrics comparisons, dynamic spend filters, and customer identifier lookups.

## Repository Contents

* `Unsupervised learning.ipynb` - Core Jupyter Notebook containing the data pipeline, model training, and analytical validations.
* `app.py` - Core production Streamlit script running the user interface web server.
* `segmented_customers_list.csv` - Data asset containing unique customer logs with engineered traits, labels, and PCA coordinates.
* `customer_segmentation_summary.csv` - Averaged behavioral trait matrix calculated for each of the 4 clusters.

## Customer Personas Discovered

* **The Premium High-Rollers**: Highest overall metrics across life value expenditures, selective premium pricing focus, and large individual cart baskets.
* **The Bulk Value Shoppers**: Focus heavily on clear unit volume discount targets; highly sensitive toward low unit costs with a higher overall drop-off latency.
* **The Fresh Prospects**: Warm recent conversion profiles showing active mid-range transactions ready for automatic onboarding cycles.
* **The Minimalist One-Timers**: Micro-basket consumers characterized by historically flat interaction traits requiring low-barrier clearing sales.

## Getting Started

### Prerequisites
Ensure your active environment contains the necessary statistical dependencies:
```bash
pip install streamlit pandas plotly scikit-learn matplotlib
```

### Relaunching the Analytical Interface
Run the Streamlit server directly from your workspace terminal:
```bash
python -m streamlit run app.py
```
The interactive interface will automatically execute and render at `http://localhost:8501`.
