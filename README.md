# Kaggle Challenge :: [Large Scale Hierarchical Text Classification](https://www.kaggle.com/c/lshtc)
[![](http://ama.liglab.fr/~partalas/lshtclogo.png)](https://www.kaggle.com/c/lshtc)

----
# LSTC 
This is an approach to classifying Wikipedia documents into one of 452167 categories without using hierarchy. The collection contains over 2 millions documents with a number of meaningful terms in each. 

#### Data
The input data files `test.csv` and `train.csv` represent document sets as vector space model and follow the format [as described](https://www.kaggle.com/c/lshtc/data).  Downloading requires logging in Kaggle.

#### Algorithm
The approach implements **k-NN** classification algorithm. Neighbours of each document from `test.csv` are searched in a subset of training instances that only contain the most meaningful terms of the processed document.
- Term weight is estimated by **TF-iDF** statistics: let ![](http://mathurl.com/hn6nmcg.png) denote *n* occurences of term *t* in document *d* in *D*, then

    ![](http://mathurl.com/jmwrmgd.png)

- Metric function of k-NN algorithm is **Jaccard distance**:

    ![](http://mathurl.com/jcuuuhd.png)
    
- Predicted categories are the most frequent categories in obtained neighbour instances. 

#### Compilation and running
- Requirements:
    - GCC 4.8 or higher;
    - 3+GB RAM;
    
- Put `test.csv` and `train.csv` into `data` folder;
- Run `make` to compile a binary `lstc`;
- Run `lstc` with

    ```
    ./lstc [cats <..>][nghbrs <..>][tfidfs <..>][threads <..>]
    ```
    Optional command line arguments are:
    - `cats` - maximum number of categories to predict, **8** by default. The number is supposed to be non-constant, the approach doesn't imply heuristics for predicting it;
    - `nghbrs` - maximum number of the nearest neighbours, **5** by default;
    - `tfidfs` - number of the most important terms to search for in the scoped document, i.e. the ones with the largest TF-iDF's, **4** by default.
    - `threads` - number of OpenMP threads.
    
The result file `output.csv` will be placed into `output` folder.