# Machine Learning Projects

1. [Supervised Learning](#supervised-learning)
    * [Decision Tree](#decision-tree)
    * [Naive Bayes](#naive-bayes-model-by-maximum-likelihood)

## Supervised Learning
#### Text Categorization

The dataset consists of a set of Reddit posts sourced from https://files.pushshift.io/reddit/. 
The dataset includes a subset of 1500 comments the r/books and r/atheism subreddits, cleaned by removing 
punctuation and some offensive language, and limiting the words to only those used more than 3 times among all posts.
These 3000 comments are split evenly into training and testing sets (with 1500 documents in each). 

The goal of the classifier is to categorize the posts/documents into either 1 = `atheism` or 2 = `books` based on
the pre-processed set of words from that post/document given as input to the classifier.

`trainData.txt`: Each line follows the format `documentId wordId`. The file contains the training data for building the decision tree.

`trainLabel.txt`: The `ith` line contains the category of the `ith` document (1 = `atheism`, 2 = `books`). The file is used for building the decision tree.

`testData.txt`: Each line follows the format `documentId wordId`. The file contains the testing data to assess the accuracy of the decision tree.
 
 `testLabel.txt`: The `ith` line contains the category of the `ith` document (1 = `atheism`, 2 = `books`). The file is used to assess the accuracy of the decision tree.

### Decision Tree

Decision tree algorithms are implemented using 2 methods:

1. Average information gain

$$ I = I(E) - [ (1/2) * I(E_1) + (1/2) * I(E_2) ] $$

2. Information gain weighted by fraction of documents on each side of the split

$$ I = I(E) - [ (N_1/N) * I(E_1) + (N_2/N) * I(E_2) ] $$

where $E$ (size $N$) is the dataset at a node, $E_1$ (size $N_1$) is the subset of the dataset with feature `F`,
$E_2$ (size $N_2$) is the subset of the dataset without feature `F` 
and $I$ is the information gain from splitting the node over feature `F`.

At each step, we choose to split the leaf with the highest information gain for its next best feature to split on.
The next best feature to split on is the feature which provides the highest information gain.

We assess the accuracy of the decision trees for the two methods of information gain calculation, 
for different number of nodes in the tree.

<img width=500 src="supervised_learning/img/avg_info_gain_accuracy.png">

<img width=500 src="supervised_learning/img/weighted_info_gain_accuracy.png">


### Naive Bayes model by Maximum Likelihood

We build the models by learning the following parameters:

* $\theta = P(category \ is \ 1)$
* $\theta_{i1} = P(word_i \ is \ present \ | category \ is \ 1)$
* $\theta_{i0} = P(word_i \ is \ present \ | category \ is \ 2)$

To account for words that occur in the test dataset but do not occur in the training dataset,
we apply a Laplace correction. For each word $i$, the parameters are learned using this formula.

$$ \theta_{i1} = ((number \ of \ documents \ with \ category \ 1 \ and \ word \ i \ present) + 1) 
                    / (number \ of \ documents \ with \ category \ 1 \ + \ 2) $$ 
                    
$$ \theta_{i0} = ((number \ of \ documents \ with \ category \ 2 and \ word \ i \ present) + 1) 
                    / ((number \ of \ documents \ with \ category \ 2) + 2) $$
                    
To learn the Naive Bayes classifier, we assume that the words are independent of each other given the category.

$$ P(Category | word_0 \ word_1 \ ... \ word_n ) \propto [ \Pi_i P(word_i | Category)] P(Category) $$

The documents are classified based on the category with the highest posterior probability $P(Category | words in document)$

Training accuracy: 91.2329%

Testing accuracy: 74.6207%
