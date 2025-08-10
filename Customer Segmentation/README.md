# 🎯 Customer Segmentation Analysis

A comprehensive customer segmentation analysis project using machine learning clustering algorithms to identify distinct customer groups based on shopping mall behavior patterns.

## 📊 Project Overview

This project analyzes customer data from a shopping mall to identify distinct customer segments using various clustering algorithms. The analysis helps businesses understand customer behavior patterns and tailor marketing strategies accordingly.

### Key Objectives:
- **Customer Grouping**: Identify distinct customer segments based on behavior patterns
- **Pattern Analysis**: Understand relationships between age, income, and spending behavior
- **Marketing Insights**: Provide actionable insights for targeted marketing campaigns
- **Algorithm Comparison**: Evaluate different clustering approaches

## 🔍 Dataset

**Mall_Customers.csv** - Contains customer information with the following features:
- **CustomerID**: Unique identifier for each customer
- **Gender**: Customer's gender (Male/Female)
- **Age**: Customer's age in years
- **Annual Income (k$)**: Annual income in thousands of dollars
- **Spending Score (1-100)**: Spending behavior score (1 = low spender, 100 = high spender)

## 🚀 Getting Started

### Prerequisites

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Running the Analysis

1. **Open the Jupyter Notebook**:
```bash
jupyter notebook Customer_Segmentation.ipynb
```

2. **Run all cells** to perform the complete customer segmentation analysis

## 🔧 Technologies Used

- **Python 3.x** - Primary programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical data visualization
- **Plotly** - Interactive visualizations
- **Scikit-learn** - Machine learning algorithms
  - K-Means Clustering
  - Hierarchical Clustering
  - DBSCAN Clustering
  - Affinity Propagation
- **SciPy** - Statistical functions and hierarchical clustering
- **Missingno** - Missing data visualization

## 📁 Project Structure

```
Customer Segmentation/
├── Customer_Segmentation.ipynb    # Main analysis notebook
├── Mall_Customers.csv            # Customer dataset
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
└── requirements.txt              # Python dependencies
```

## 🔬 Analysis Methods

### 1. Data Preprocessing
- **Data Cleaning**: Handle missing values and duplicates
- **Feature Engineering**: Prepare features for clustering
- **Data Scaling**: Normalize numerical features using StandardScaler
- **Encoding**: Convert categorical variables to numerical format

### 2. Clustering Algorithms

#### K-Means Clustering
- **Purpose**: Partition customers into K distinct groups
- **Method**: Iterative optimization of cluster centroids
- **Evaluation**: Elbow method and silhouette analysis

#### Hierarchical Clustering
- **Purpose**: Create a tree-like structure of customer relationships
- **Method**: Agglomerative clustering with dendrogram visualization
- **Advantage**: No need to specify number of clusters beforehand

#### DBSCAN Clustering
- **Purpose**: Density-based clustering for irregular cluster shapes
- **Method**: Groups points based on density connectivity
- **Parameters**: Epsilon (neighborhood radius) and MinPts

#### Affinity Propagation
- **Purpose**: Automatic determination of optimal number of clusters
- **Method**: Message passing between data points
- **Advantage**: Self-tuning clustering algorithm

### 3. Evaluation Metrics
- **Silhouette Score**: Measures cluster quality and separation
- **Elbow Method**: Determines optimal number of clusters
- **Dendrogram Analysis**: Visual assessment of hierarchical structure

## 📈 Key Insights

### Customer Segments Identified:
1. **High Income, High Spending**: Premium customers with high purchasing power
2. **High Income, Low Spending**: Conservative high-earners
3. **Low Income, High Spending**: Young customers with high spending tendency
4. **Low Income, Low Spending**: Budget-conscious customers
5. **Average Income, Average Spending**: Mainstream customers

### Business Applications:
- **Targeted Marketing**: Customize campaigns for each segment
- **Product Placement**: Optimize store layout based on customer behavior
- **Pricing Strategy**: Develop segment-specific pricing models
- **Customer Retention**: Design loyalty programs for high-value segments

## 🎨 Visualizations

The project includes comprehensive visualizations:
- **Scatter Plots**: Customer distribution by income vs. spending
- **Box Plots**: Statistical distribution of features by segments
- **Dendrograms**: Hierarchical clustering relationships
- **3D Plots**: Multi-dimensional customer clustering
- **Correlation Heatmaps**: Feature relationships analysis

## 📝 Usage Examples

### Basic Clustering:
```python
# K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
customer_segments = kmeans.fit_predict(scaled_features)

# Hierarchical clustering
hierarchical = AgglomerativeClustering(n_clusters=5)
hierarchical_segments = hierarchical.fit_predict(scaled_features)
```

### Evaluation:
```python
# Silhouette score
silhouette_avg = silhouette_score(scaled_features, customer_segments)

# Elbow method
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertias.append(kmeans.inertia_)
```

## 🔧 Customization

### Adding New Features:
1. Include additional customer attributes in the dataset
2. Modify preprocessing steps in the notebook
3. Adjust clustering parameters for optimal results

### Algorithm Tuning:
- **K-Means**: Adjust n_clusters and random_state
- **DBSCAN**: Tune epsilon and min_samples parameters
- **Hierarchical**: Choose linkage method (ward, complete, average)

## 📊 Performance Considerations

- **Dataset Size**: Current analysis handles 200+ customers efficiently
- **Scalability**: Algorithms can be extended to larger datasets
- **Computational Cost**: K-Means is fastest, Hierarchical is most interpretable
- **Memory Usage**: Optimized for typical customer segmentation datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the clustering algorithms
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset source: Mall Customer Segmentation Data
- Scikit-learn community for clustering algorithms
- Python data science ecosystem

---

**Built with ❤️ using Python and Machine Learning for Customer Insights**
