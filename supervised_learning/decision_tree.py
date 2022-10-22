import pandas as pd
import heapq
from typing import List, Optional
from anytree import Node, RenderTree
import numpy as np
import matplotlib.pyplot as plt

'''
testData and trainData: docId wordId
trainLabel and testLabel: categories(target) (docId = line #) - always deduct one in the code since indexing starts at 0
words: words (wordId = line number) - always deduct one in the code since indexing starts at 0
'''

CATEGORIES = {
    1: "atheism",
    2: "books",
}

TREE = "tree"
PQ = "pq"
INODE_COUNT = "inode_count"


class TreeNode:
    def __init__(self, dataset, point_est, best_next_feature_id, info_gain, used_features_split):
        self.dataset = dataset
        self.point_est = point_est
        self.best_next_feature_id = best_next_feature_id
        self.info_gain = info_gain
        self.used_features_split = used_features_split
        self.child_with_feature = None
        self.child_without_feature = None

    def __lt__(self, other):
        return self.info_gain > other.info_gain


class Dataset:
    def __init__(self, size: int, category1_count: int, category2_count: int):
        self.size = size  # number of unique docIds
        self.category1_count = category1_count  # number of docIds in category 1
        self.category2_count = category2_count  # number of docIds in category 2


# *******************************************************
# Import the data
# *******************************************************
def prep_data():
    def data_to_list(filename):
        with open(filename) as f:
            lst = [line.rstrip() for line in f]
        return lst

    train_data = pd.read_csv("train_data/trainData.txt", delimiter=" ", header=None)
    train_label = pd.read_csv("train_data/trainLabel.txt", header=None)

    test_data = pd.read_csv("test_data/testData.txt", delimiter=" ", header=None)
    test_label = pd.read_csv("test_data/testLabel.txt", header=None)
    words = data_to_list("words.txt")

    train_data.columns = ["docId", "wordId"]
    train_label.columns = ["category"]
    test_data.columns = ["docId", "wordId"]
    test_label.columns = ["category"]

    return train_data, train_label, test_data, test_label, words


# *******************************************************
# Helper functions to build decision tree
# *******************************************************

def split(data, wordId):
    docId_with_target = np.unique(data[data.wordId == wordId]["docId"])
    data_with_target = data[data["docId"].isin(docId_with_target)]
    data_without_target = data[~data["docId"].isin(docId_with_target)]
    return data_with_target, data_without_target


def get_node_attr(train_data, label, is_weighted_info_gain: bool, used_features):
    data = train_data.copy()
    data['indicator'] = 1
    pivot_table = data.pivot(index='docId', columns='wordId', values='indicator')
    pivot_table = pivot_table.loc[:, ~pivot_table.columns.isin(used_features)]
    pivot_table = pivot_table.fillna(0)
    label["docId"] = [x + 1 for x in range(len(label))]
    merge_df = pd.merge(pivot_table, label, how="left", on=["docId"])

    #####
    # Finding point estimate
    category_1_count = len(merge_df[merge_df.category == 1])
    category_2_count = len(merge_df[merge_df.category == 2])
    if category_1_count >= category_2_count:
        point_est = 1
    else:
        point_est = 2
    ####

    # Finding next best feature

    df_inversed = pivot_table
    df_inversed = df_inversed.replace(1, 4)
    df_inversed = df_inversed.replace(0, 1)
    df_inversed = df_inversed.replace(4, 0)
    merge_df_inversed = pd.merge(df_inversed, label, how="left", on=["docId"])

    total_doc = len(pivot_table)

    feature_stat = pd.DataFrame(columns=['feature'])
    feature_stat['feature'] = pivot_table.columns
    feature_stat['with_feature'] = pivot_table.sum().values
    feature_stat['without_feature'] = total_doc - feature_stat['with_feature']
    feature_stat['with_f_cat_1'] = merge_df[merge_df['category'] == 1].iloc[:, 1:-1].sum().values
    feature_stat['with_f_cat_2'] = merge_df[merge_df['category'] == 2].iloc[:, 1:-1].sum().values
    feature_stat['wo_f_cat_1'] = merge_df_inversed[merge_df_inversed['category'] == 1].iloc[:, 1:-1].sum().values
    feature_stat['wo_f_cat_2'] = merge_df_inversed[merge_df_inversed['category'] == 2].iloc[:, 1:-1].sum().values
    feature_stat['category_1'] = feature_stat['with_f_cat_1'] + feature_stat['wo_f_cat_1']
    feature_stat['category_2'] = feature_stat['with_f_cat_2'] + feature_stat['wo_f_cat_2']

    total_features = len(feature_stat)
    feature_stat['info_content_total'] = np.zeros(total_features)
    feature_stat["info_content_w_f"] = np.zeros(total_features)
    feature_stat['info_content_wo_f'] = np.zeros(total_features)

    # Calculate category 1 portion of info_content
    feature_stat.loc[feature_stat.category_1 != 0, 'info_content_total'] = \
        -((feature_stat[feature_stat.category_1 != 0]['category_1'] / total_doc) *
          np.log2(feature_stat[feature_stat.category_1 != 0]['category_1'] / total_doc))
    feature_stat.loc[feature_stat.with_f_cat_1 != 0, "info_content_w_f"] = \
        -((feature_stat[feature_stat.with_f_cat_1 != 0]['with_f_cat_1'] / feature_stat[feature_stat.with_f_cat_1 != 0][
            'with_feature']) *
          np.log2(feature_stat[feature_stat.with_f_cat_1 != 0]['with_f_cat_1'] /
                  feature_stat[feature_stat.with_f_cat_1 != 0]['with_feature']))
    feature_stat.loc[feature_stat.wo_f_cat_1 != 0, 'info_content_wo_f'] = \
        -((feature_stat[feature_stat.wo_f_cat_1 != 0]['wo_f_cat_1'] / feature_stat[feature_stat.wo_f_cat_1 != 0][
            'without_feature']) *
          np.log2(feature_stat[feature_stat.wo_f_cat_1 != 0]['wo_f_cat_1'] / feature_stat[feature_stat.wo_f_cat_1 != 0][
              'without_feature']))

    # Add category 2 portion of info_content
    feature_stat.loc[feature_stat.category_2 != 0, 'info_content_total'] += \
        -((feature_stat[feature_stat.category_2 != 0]['category_2'] / total_doc) *
          np.log2(feature_stat[feature_stat.category_2 != 0]['category_2'] / total_doc))
    feature_stat.loc[feature_stat.with_f_cat_2 != 0, "info_content_w_f"] += \
        -((feature_stat[feature_stat.with_f_cat_2 != 0]['with_f_cat_2'] / feature_stat[feature_stat.with_f_cat_2 != 0][
            'with_feature']) *
          np.log2(feature_stat[feature_stat.with_f_cat_2 != 0]['with_f_cat_2'] /
                  feature_stat[feature_stat.with_f_cat_2 != 0]['with_feature']))
    feature_stat.loc[feature_stat.wo_f_cat_2 != 0, 'info_content_wo_f'] += \
        -((feature_stat[feature_stat.wo_f_cat_2 != 0]['wo_f_cat_2'] / feature_stat[feature_stat.wo_f_cat_2 != 0][
            'without_feature']) *
          np.log2(feature_stat[feature_stat.wo_f_cat_2 != 0]['wo_f_cat_2'] / feature_stat[feature_stat.wo_f_cat_2 != 0][
              'without_feature']))

    if is_weighted_info_gain:
        # info_original - ((split1_size / original_size) * info_split1 + (split2_size / original_size) * info_split2)
        feature_stat['info_gain'] = feature_stat['info_content_total'] - (
                (feature_stat['with_feature'] / total_doc) * feature_stat["info_content_w_f"] +
                (feature_stat['without_feature'] / total_doc) * feature_stat["info_content_wo_f"]
        )
    else:
        # info_original - (0.5 * info_split1 + 0.5 * info_split2)
        feature_stat['info_gain'] = feature_stat['info_content_total'] - (
                0.5 * feature_stat["info_content_w_f"] + 0.5 * feature_stat["info_content_wo_f"]
        )

    best_feature = feature_stat[feature_stat.info_gain == feature_stat.info_gain.max()]
    if best_feature.empty:
        return 0, 0, point_est
    return best_feature['feature'].values[0], best_feature['info_gain'].values[0], point_est



def create_node(data, label, is_weighted_info_gain, used_features_split):
    wordId_of_best, info_gain, pt_est = get_node_attr(data, label, is_weighted_info_gain, used_features_split)
    return TreeNode(data, pt_est, wordId_of_best, info_gain, used_features_split)


# *******************************************************
# Build decision tree
# *******************************************************
def decision_tree_learner(
        target_features: pd.DataFrame, train_data: pd.DataFrame, is_weighted_info_gain: bool, node_limit: int,
        tree: Optional[TreeNode], pq: List, internal_node_count: int,
):
    '''
    input_features: words
    target_features: labels/categories
    examples: train_data
    returns the root of the tree, the pq built and the internal node count
    '''
    root = tree
    if tree is None:
        if internal_node_count != 0 or pq != []:
            print("pq should be empty and node count should be zero")
        root = create_node(train_data, target_features, is_weighted_info_gain, [])
        heapq.heappush(pq, (-root.info_gain, root))
    # heapq is for min heap, since the information gain is never negative,
    # i am adding the node with the negative of the information gain
    # so it will behave similar to a max heap with the actual (positive) information gain
    node_count = internal_node_count
    while node_count < node_limit:
        node_count += 1
        if not pq:
            break
        next_split: TreeNode = heapq.heappop(pq)[1]
        split1, split2 = split(next_split.dataset, next_split.best_next_feature_id)
        node_with_target = create_node(
            split1, target_features, is_weighted_info_gain,
            next_split.used_features_split + [next_split.best_next_feature_id]
        )
        node_without_target = create_node(
            split2, target_features, is_weighted_info_gain,
            next_split.used_features_split + [next_split.best_next_feature_id]
        )
        next_split.child_with_feature = node_with_target
        next_split.child_without_feature = node_without_target

        heapq.heappush(pq, (-node_with_target.info_gain, node_with_target))
        heapq.heappush(pq, (-node_without_target.info_gain, node_without_target))
    return root, pq, node_count


# *******************************************************
# Print tree with 10 nodes
# *******************************************************
def build_any_tree(decision_tree_node: TreeNode, with_feature: bool, parent: Node, input_features):
    if decision_tree_node.child_with_feature is None:
        Node(
            decision_tree_node.best_next_feature_id,
            split_feature=input_features[decision_tree_node.best_next_feature_id],
            parent=parent,
            child_has_previous_split_feature=with_feature,
            prediction=CATEGORIES[decision_tree_node.point_est],
        )
        return

    n = Node(
        decision_tree_node.best_next_feature_id,
        parent=parent,
        split_feature=input_features[decision_tree_node.best_next_feature_id],
        child_has_previous_split_feature=with_feature,
        info_gain=round(decision_tree_node.info_gain, 4),
    )
    build_any_tree(decision_tree_node.child_with_feature, True, n, input_features)
    build_any_tree(decision_tree_node.child_without_feature, False, n, input_features)


def decision_tree_printout(train_data, train_label, words):
    '''
    Print decision tree for average information gain selection method
    '''

    decision_tree, _, _ = decision_tree_learner(train_label, train_data, False, 10, None, [], 0)
    tree_printout = Node(
        decision_tree.best_next_feature_id,
        split_feature=words[decision_tree.best_next_feature_id],
        info_gain=decision_tree.info_gain,
    )
    build_any_tree(decision_tree.child_with_feature, True, tree_printout, words)
    build_any_tree(decision_tree.child_without_feature, False, tree_printout, words)
    print("\n")
    print("Decision tree using average information gain as selection method")
    print(RenderTree(tree_printout))
    print("*" * 15)

    decision_tree, _, _ = decision_tree_learner(train_label, train_data, True, 10, None, [], 0)
    tree_printout = Node(
        decision_tree.best_next_feature_id,
        split_feature=words[decision_tree.best_next_feature_id],
        info_gain=round(decision_tree.info_gain, 4),
    )
    build_any_tree(decision_tree.child_with_feature, True, tree_printout, words)
    build_any_tree(decision_tree.child_without_feature, False, tree_printout, words)
    print("\n")
    print("Decision tree using weighted information gain as selection method")
    print(RenderTree(tree_printout))
    print("*" * 15)


# *******************************************************
# Get accuracy from using train data vs test data
# as the number of nodes increases
# *******************************************************

def classify_doc(doc_words, decision_tree):
    s: TreeNode = decision_tree
    while s.child_with_feature:
        decision_feature = s.best_next_feature_id
        if decision_feature in doc_words['wordId'].values:
            s = s.child_with_feature
        else:
            s = s.child_without_feature
    return s.point_est


def get_accuracy(test_doc_ids, test_doc_count, train_docs_ids, train_doc_count, train_data, train_label, test_data, test_label, tree_size, avg_ig,
                 weighted_ig):
    weighted_ig[TREE], weighted_ig[PQ], weighted_ig[INODE_COUNT] = decision_tree_learner(
        train_label, train_data, True, tree_size, weighted_ig[TREE], weighted_ig[PQ], weighted_ig[INODE_COUNT]
    )
    avg_ig[TREE], avg_ig[PQ], avg_ig[INODE_COUNT] = decision_tree_learner(
        train_label, train_data, False, tree_size, avg_ig[TREE], avg_ig[PQ], avg_ig[INODE_COUNT]
    )
    correctly_classified_weighted_test = 0
    correctly_classified_avg_test = 0
    for doc in test_doc_ids:
        data_seg = test_data[test_data["docId"] == doc]
        weighted_if_prediction = classify_doc(data_seg, weighted_ig[TREE])
        avg_if_prediction = classify_doc(data_seg, avg_ig[TREE])
        actual_classification = test_label.loc[doc - 1, 'category']
        if weighted_if_prediction == actual_classification:
            correctly_classified_weighted_test += 1
        if avg_if_prediction == actual_classification:
            correctly_classified_avg_test += 1

    correctly_classified_weighted_train = 0
    correctly_classified_avg_train = 0
    for doc in train_docs_ids:
        data_seg = train_data[train_data["docId"] == doc]
        weighted_if_prediction = classify_doc(data_seg, weighted_ig[TREE])
        avg_if_prediction = classify_doc(data_seg, avg_ig[TREE])
        actual_classification = train_label.loc[doc - 1, 'category']
        if weighted_if_prediction == actual_classification:
            correctly_classified_weighted_train += 1
        if avg_if_prediction == actual_classification:
            correctly_classified_avg_train += 1

    return (correctly_classified_avg_test * 100.0) / test_doc_count, \
           (correctly_classified_weighted_test * 100.0) / test_doc_count, \
           (correctly_classified_avg_train * 100.0) / train_doc_count, \
           (correctly_classified_weighted_train * 100.0) / train_doc_count, \
           avg_ig, weighted_ig


def generate_graph(df, x, y_test, y_train, selection_type):
    plt.plot(df[x], df[y_test], color='r', label="Test Accuracy")
    plt.plot(df[x], df[y_train], color='g', label="Train Accuracy")
    plt.xlabel('Number of nodes in decision tree')
    plt.ylabel('Percentage accuracy (%)')
    plt.title(f"Training and testing accuracy using {selection_type}")
    plt.legend()
    plt.show()


def build_accuracy_report(train_data, train_label, test_data, test_label):
    test_docs = np.unique(test_data["docId"])
    test_doc_count = len(test_docs)
    train_docs = np.unique(train_data["docId"])
    train_doc_count = len(train_docs)
    report_df = pd.DataFrame(columns=[
        'nodeCount', 'avg_ig_accuracy_test', 'weighted_ig_accuracy_test',
        'avg_ig_accuracy_train', 'weighted_ig_accuracy_train'
    ])

    weighted_ig_detail = {
        TREE: None,
        PQ: [],
        INODE_COUNT: 0,
    }

    avg_ig_detail = {
        TREE: None,
        PQ: [],
        INODE_COUNT: 0,
    }

    for tree_size in range(1, 101):
        avg_ig_acc_test, weighted_ig_acc_test, avg_ig_acc_train, weighted_ig_acc_train, avg_ig_detail, weighted_ig_detail = get_accuracy(
            test_docs, test_doc_count, train_docs, train_doc_count, train_data, train_label, test_data, test_label,
            tree_size, avg_ig_detail, weighted_ig_detail,
        )
        report_df = report_df.append(
            {'nodeCount': tree_size,
             'avg_ig_accuracy_test': avg_ig_acc_test,
             'weighted_ig_accuracy_test': weighted_ig_acc_test,
             'avg_ig_accuracy_train': avg_ig_acc_train,
             'weighted_ig_accuracy_train': weighted_ig_acc_train
             },
            ignore_index=True
        )
    generate_graph(
        report_df, 'nodeCount', 'avg_ig_accuracy_test', 'avg_ig_accuracy_train', "Average Info Gain Feature Selection"
    )
    generate_graph(
        report_df, 'nodeCount', 'weighted_ig_accuracy_test', 'weighted_ig_accuracy_train', "Weighted Info Gain Feature Selection"
    )


def main():
    train_data, train_label, test_data, test_label, words = prep_data()
    decision_tree_printout(train_data, train_label, words)
    build_accuracy_report(train_data, train_label, test_data, test_label)


if __name__ == "__main__":
    main()
