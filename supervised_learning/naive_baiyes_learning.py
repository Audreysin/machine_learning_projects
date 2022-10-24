import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 20)

'''
testData and trainData: docId wordId
trainLabel and testLabel: categories(target) (docId = line #) - always deduct one in the code since indexing starts at 0
words: words (wordId = line number) - always deduct one in the code since indexing starts at 0


CATEGORIES = {
    1: "atheism",
    2: "books",
}


The predict class C:
    C is true if category 1
    C is false if category 2
'''


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


def clean_data(t_data, label):
    '''
    :param t_data: data dataframe; columns["docId" , "wordId"]
    :param label:  label dataframe; columns ["category"]
    :return:
    dataframe with the first column being the docId,
    then one column for each wordId in train_data,
    ending with a column containing the category
    '''

    data = t_data.copy()
    data['indicator'] = 1
    pivot_table = data.pivot(index='docId', columns='wordId', values='indicator')
    pivot_table = pivot_table.fillna(0)
    label["docId"] = [x + 1 for x in range(len(label))]
    merge_df = pd.merge(pivot_table, label, how="left", on=["docId"])
    return merge_df


def compute_params(data, feature_count):
    '''
    :param data: cleaned data from clean_data
    :param feature_count: number of features that exist
    :return:
    '''
    docs_count = len(data)
    category_1_count = len(data[data.category == 1])
    category_2_count = docs_count - category_1_count

    prob_category_1 = (category_1_count * 1.0) / docs_count
    word_ids = [x + 1 for x in range(feature_count)]
    tuples = []  # a tuple will contain (wordId, theta_i1, theta_i0)

    for word_id in word_ids:
        if word_id in data:
            count_have_wordId_and_cat1 = len(data[(data[word_id] == 1) & (data.category == 1)])
            count_have_wordId_and_cat2 = len(data[(data[word_id] == 1) & (data.category == 2)])
        else:
            count_have_wordId_and_cat1 = 0
            count_have_wordId_and_cat2 = 0

        prob_wordId_given_cat_1 = (count_have_wordId_and_cat1 + 1.0) / (category_1_count + 2.0)  # P(A_i=true | C=true)
        prob_wordId_given_cat_2 = (count_have_wordId_and_cat2 + 1.0) / (category_2_count + 2.0)  # P(A_i=true | C=false)

        tuples.append((word_id, prob_wordId_given_cat_1, prob_wordId_given_cat_2))

    attribute_theta = pd.DataFrame(tuples, columns=[
        'wordId', 'prob_feature_given_C_true', 'prob_feature_given_C_false'
    ])

    return prob_category_1, attribute_theta


def print_most_discriminative_features(theta_i, words, limit):
    attribute_theta = theta_i.copy()
    attribute_theta["discriminative_score"] = np.absolute(np.log2(attribute_theta['prob_feature_given_C_true']) - \
                                                          np.log2(attribute_theta['prob_feature_given_C_false']))
    most_discr_features = attribute_theta.sort_values(
        'discriminative_score', ascending=False
    ).head(limit)
    print(f"{limit} most discriminative features")
    for row in most_discr_features.values:
        feature = row[0]
        discriminative_score = round(row[3], 5)
        print(f"{words[int(feature - 1)]}  {feature}  {discriminative_score}")
    print("***************************************")
    print("\n")
    return


def get_accuracy(data_summarized, theta, theta_i):
    data_adj_cat1 = data_summarized.copy()
    data_adj_cat2 = data_summarized.copy()  # dataframe will be filled
    data_adj_cat1 = data_adj_cat1.replace(0, np.nan)
    data_adj_cat2 = data_adj_cat2.replace(0, np.nan)
    data_adj_cat1["posterior_prob_numerator"] = 1
    data_adj_cat2["probability_product"] = 1

    for col in data_adj_cat1:
        if col != "docId" and col != "category" and col != "posterior_prob_numerator":
            # Then col is the wordId
            prob = theta_i[theta_i.wordId == col]["prob_feature_given_C_true"].values[0]
            data_adj_cat1[col] = data_adj_cat1[col].replace(1, prob)
            data_adj_cat1[col] = data_adj_cat1[col].replace(np.nan, 1-prob)
            data_adj_cat1["posterior_prob_numerator"] *= data_adj_cat1[col]

    for col in data_adj_cat2:
        if col != "docId" and col != "category" and col != "probability_product":
            # then col is wordId
            prob = theta_i[theta_i.wordId == col]["prob_feature_given_C_false"].values[0]
            data_adj_cat2[col] = data_adj_cat2[col].replace(1, prob)
            data_adj_cat2[col] = data_adj_cat2[col].replace(np.nan, 1-prob)
            data_adj_cat2["probability_product"] *= data_adj_cat2[col]

    data_adj_cat1["posterior_prob_numerator"] *= theta
    data_adj_cat1["posterior_prob_denominator"] = (data_adj_cat1["posterior_prob_numerator"]) + \
                                                  (data_adj_cat2["probability_product"] * (1 - theta))
    data_adj_cat1["posterior_prob"] = data_adj_cat1["posterior_prob_numerator"] / \
                                      data_adj_cat1["posterior_prob_denominator"]
    data_adj_cat1["predicted_category"] = 2
    data_adj_cat1.loc[data_adj_cat1['posterior_prob'] > 0.5, 'predicted_category'] = 1
    data_adj_cat1["prediction_accurate"] = data_adj_cat1.apply(
        lambda x: True if x['predicted_category'] == x["category"] else False, axis=1
    )
    prediction_count = len(data_adj_cat1)
    accurate_prediction_count = len(data_adj_cat1[data_adj_cat1.prediction_accurate == True])
    return (accurate_prediction_count * 100.0) / prediction_count


def main():
    train_data, train_label, test_data, test_label, words = prep_data()
    train_data_cleaned = clean_data(train_data, train_label)
    test_data_cleaned = clean_data(test_data, test_label)
    theta, theta_i = compute_params(train_data_cleaned, len(words))
    print_most_discriminative_features(theta_i, words, 10)
    train_accuracy = get_accuracy(train_data_cleaned, theta, theta_i)
    test_accuracy = get_accuracy(test_data_cleaned, theta, theta_i)
    print(f"Training accuracy: {train_accuracy}%")
    print(f"Testing accuracy: {test_accuracy}%")


if __name__ == "__main__":
    main()
