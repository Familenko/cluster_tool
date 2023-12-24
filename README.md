# Сluster tool

## Description

Cluster Tool is a user-friendly solution that expedites the 
clustering process. It provides a simple interface for creating 
clustered data and models, supporting popular algorithms such 
as KMeans, DBSCAN, and AgglomerativeClustering. This tool is 
specifically designed to accelerate and simplify the clustering 
experience, making it accessible for beginners in data science.

## Requirements

* Python 3.10+
* Jupyter Notebook

## Installation

```
git clone https://github.com/Familenko/cluster_tool.git

cd path.to.cluster_tool

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

## Usage

```
from cluster_tool import Cluster
cl = Cluster(df)
```

Detailed description of the methods and 
examples of use can be found in the [notebook](test_notebook.ipynb)
