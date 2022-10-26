import pandas as pd
import numpy as np
from numpy.random import uniform as rand
import statistics
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 100)

SLOEPNEA = "sloepnea"
FORENNDITIS = "forenditis"
DEGAR = "degar"
DUNETTS = "dunetts"
TGENE = "tGene"
PRESENT = 1
ABSENT = 0
MILD = 1
SEVERE = 2
NOT_OBSERVED = -1
PROB = "probability"
DUNETTS_NOTE = "dunetts_note"
P_FORENNDITIS_GIVEN_DUNETTS = "p_fo_given_dun"
P_DEGAR_GIVEN_DUNETTS = "p_degar_given_dun"
P_SLOEPNEA_GIVEN_GENES_DUNETTS = "p_slo_given_genes_dun"
P_DUNETTS = "p_dun"
MAX_LIKELIHOOD = "max_likelihood"
PREDICTED = "predicted"
P_DUNETTS_GIVEN_EVIDENCE = P_DUNETTS + "_given_evidence"
P_GENES = "p_genes"
NOT_OBSERVED_STR = "NOT_OBSERVED"
OBSERVED_ACTUAL = "OBSERVED_ACTUAL"
OBSERVED_NOT_POSSIBLE = "OBSERVED_NOT_POSSIBLE"

def import_data(file_name):
    cols = [SLOEPNEA, FORENNDITIS, DEGAR, TGENE, DUNETTS]
    with open(file_name) as f:
        data = []
        for line in f:
            row = [int(x) for x in line.lstrip().split()]
            data.append(row)
    df = pd.DataFrame(data, columns=cols)
    return df


def generate_initial_CPT(delta):
    # P(Dunetts)
    delta_1 = rand(0, delta)
    delta_2 = rand(0, delta)
    delta_3 = rand(0, delta)
    denominator_123 = 1 + delta_1 + delta_2 + delta_3
    tuples_0 = [(ABSENT, (0.5 + delta_1) / denominator_123), (MILD, (0.25 + delta_2) / denominator_123),
                (SEVERE, (0.25 + delta_3) / denominator_123)]
    p_dunnetts = pd.DataFrame(tuples_0, columns=[DUNETTS, PROB])

    # P(Forennditis | Dunetts)
    delta_4 = rand(0, delta)
    delta_5 = rand(0, delta)
    denominator_45 = 1 + delta_4 + delta_5
    delta_6 = rand(0, delta)
    delta_7 = rand(0, delta)
    denominator_67 = 1 + delta_6 + delta_7
    delta_8 = rand(0, delta)
    delta_9 = rand(0, delta)
    denominator_89 = 1 + delta_8 + delta_9
    tuples_1 = [(PRESENT, ABSENT, (0.001 + delta_4) / denominator_45),
                (ABSENT, ABSENT, (0.999 + delta_5) / denominator_45),
                (PRESENT, MILD, (0.99 + delta_6) / denominator_67), (ABSENT, MILD, (0.01 + delta_7) / denominator_67),
                (PRESENT, SEVERE, (0.1 + delta_8) / denominator_89), (ABSENT, SEVERE, (0.9 + delta_9) / denominator_89)]
    p_forennditis_dunnetts = pd.DataFrame(tuples_1, columns=[FORENNDITIS, DUNETTS, PROB])

    # P(Degar | Dunetts)
    delta_4 = rand(0, delta)
    delta_5 = rand(0, delta)
    denominator_45 = 1 + delta_4 + delta_5
    delta_6 = rand(0, delta)
    delta_7 = rand(0, delta)
    denominator_67 = 1 + delta_6 + delta_7
    delta_8 = rand(0, delta)
    delta_9 = rand(0, delta)
    denominator_89 = 1 + delta_8 + delta_9
    tuples_2 = [(PRESENT, ABSENT, (0.005 + delta_4) / denominator_45),
                (ABSENT, ABSENT, (0.995 + delta_5) / denominator_45),
                (PRESENT, MILD, (0.2 + delta_6) / denominator_67), (ABSENT, MILD, (0.8 + delta_7) / denominator_67),
                (PRESENT, SEVERE, (0.95 + delta_8) / denominator_89),
                (ABSENT, SEVERE, (0.05 + delta_9) / denominator_89)]
    p_degar_dunnetts = pd.DataFrame(tuples_2, columns=[DEGAR, DUNETTS, PROB])

    # P(Sloepnea | Dunetts)
    delta_0 = rand(0, delta)
    delta_1 = rand(0, delta)
    denominator_01 = 1 + delta_1 + delta_0
    delta_2 = rand(0, delta)
    delta_3 = rand(0, delta)
    denominator_23 = 1 + delta_2 + delta_3
    delta_4 = rand(0, delta)
    delta_5 = rand(0, delta)
    denominator_45 = 1 + delta_4 + delta_5
    delta_6 = rand(0, delta)
    delta_7 = rand(0, delta)
    denominator_67 = 1 + delta_6 + delta_7
    delta_8 = rand(0, delta)
    delta_9 = rand(0, delta)
    denominator_89 = 1 + delta_8 + delta_9
    delta_10 = rand(0, delta)
    delta_11 = rand(0, delta)
    denominator_1011 = 1 + delta_10 + delta_11
    tuples_3 = [(PRESENT, ABSENT, PRESENT, (0.001 + delta_0) / denominator_01),
                (ABSENT, ABSENT, PRESENT, (0.999 + delta_1) / denominator_01),
                (PRESENT, MILD, PRESENT, (0.01 + delta_2) / denominator_23),
                (ABSENT, MILD, PRESENT, (0.99 + delta_3) / denominator_23),
                (PRESENT, SEVERE, PRESENT, (0.01 + delta_4) / denominator_45),
                (ABSENT, SEVERE, PRESENT, (0.99 + delta_5) / denominator_45),
                (PRESENT, ABSENT, ABSENT, (0.01 + delta_6) / denominator_67),
                (ABSENT, ABSENT, ABSENT, (0.99 + delta_7) / denominator_67),
                (PRESENT, MILD, ABSENT, (0.8 + delta_8) / denominator_89),
                (ABSENT, MILD, ABSENT, (0.2 + delta_9) / denominator_89),
                (PRESENT, SEVERE, ABSENT, (0.85 + delta_10) / denominator_1011),
                (ABSENT, SEVERE, ABSENT, (0.15 + delta_11) / denominator_1011)]
    p_sloepnea_dunnetts_genes = pd.DataFrame(tuples_3, columns=[SLOEPNEA, DUNETTS, TGENE, PROB])

    # P(Genes)
    delta_0 = rand(0, delta)
    delta_1 = rand(0, delta)
    denominator_01 = 1 + delta_1 + delta_0
    tuples_4 = [(PRESENT, (0.1 + delta_0) / denominator_01), (ABSENT, (0.9 + delta_1) / denominator_01)]
    p_genes = pd.DataFrame(tuples_4, columns=[TGENE, PROB])

    return p_dunnetts, p_forennditis_dunnetts, p_degar_dunnetts, p_sloepnea_dunnetts_genes, p_genes


def get_prob_dun_given_evidence(data, p_dun, p_fo_dunnetts, p_degar_dun, p_slo_dun_genes, p_genes):
    for dunnetts_form in [ABSENT, MILD, SEVERE]:
        temp = p_fo_dunnetts[p_fo_dunnetts[DUNETTS] == dunnetts_form].copy()
        del temp[DUNETTS]
        temp.rename(columns={PROB: P_FORENNDITIS_GIVEN_DUNETTS + str(dunnetts_form)}, inplace=True)
        data = pd.merge(data, temp, how="inner", on=[FORENNDITIS])

        temp = p_degar_dun[p_degar_dun[DUNETTS] == dunnetts_form].copy()
        del temp[DUNETTS]
        temp.rename(columns={PROB: P_DEGAR_GIVEN_DUNETTS + str(dunnetts_form)}, inplace=True)
        data = pd.merge(data, temp, how="inner", on=[DEGAR])

        temp = p_slo_dun_genes[p_slo_dun_genes[DUNETTS] == dunnetts_form].copy()
        del temp[DUNETTS]
        temp.rename(columns={PROB: P_SLOEPNEA_GIVEN_GENES_DUNETTS + str(dunnetts_form)}, inplace=True)
        data = pd.merge(data, temp, how="inner", on=[SLOEPNEA, TGENE])

        data[P_DUNETTS + str(dunnetts_form)] = p_dun[p_dun[DUNETTS] == dunnetts_form][PROB].values[0]

    data[P_GENES] = p_genes[p_genes[TGENE] == ABSENT][PROB].values[0]
    data.loc[data[TGENE] == PRESENT, P_GENES] = p_genes[p_genes[TGENE] == PRESENT][PROB].values[0]

    for dunnetts_form in [ABSENT, MILD, SEVERE]:
        data[P_DUNETTS + str(dunnetts_form) + "_and_evidence"] = \
            data[P_FORENNDITIS_GIVEN_DUNETTS + str(dunnetts_form)] * \
            data[P_DEGAR_GIVEN_DUNETTS + str(dunnetts_form)] * \
            data[P_SLOEPNEA_GIVEN_GENES_DUNETTS + str(dunnetts_form)] * \
            data[P_GENES] * \
            data[P_DUNETTS + str(dunnetts_form)]

    return data


def get_accuracy(test_data, p_dun, p_fo_dunnetts, p_degar_dun, p_slo_dun_genes, p_genes):
    probabilities = get_prob_dun_given_evidence(test_data, p_dun, p_fo_dunnetts, p_degar_dun, p_slo_dun_genes, p_genes)
    probabilities[MAX_LIKELIHOOD] = probabilities[
        [P_DUNETTS + "0_and_evidence", P_DUNETTS + "1_and_evidence", P_DUNETTS + "2_and_evidence"]].max(axis=1)
    probabilities[PREDICTED] = 0
    probabilities.loc[probabilities[P_DUNETTS + "1_and_evidence"] == probabilities[MAX_LIKELIHOOD], PREDICTED] = 1
    probabilities.loc[probabilities[P_DUNETTS + "2_and_evidence"] == probabilities[MAX_LIKELIHOOD], PREDICTED] = 2
    correct_predictions = len(probabilities[probabilities[DUNETTS] == probabilities[PREDICTED]])
    total_data = len(probabilities)
    return correct_predictions * 100.0 / total_data


def augment_data(train_data):
    data_with_dunetts_observed_first = train_data[train_data[DUNETTS] != NOT_OBSERVED].copy()
    data_with_dunetts_observed_first[DUNETTS_NOTE] = OBSERVED_ACTUAL
    data_with_dunetts_observed_second = data_with_dunetts_observed_first.copy()
    data_with_dunetts_observed_third = data_with_dunetts_observed_first.copy()

    data_with_dunetts_observed_first.loc[data_with_dunetts_observed_first[DUNETTS] != ABSENT, DUNETTS_NOTE] = OBSERVED_NOT_POSSIBLE
    data_with_dunetts_observed_first[DUNETTS] = ABSENT
    data_with_dunetts_observed_second.loc[data_with_dunetts_observed_second[DUNETTS] != MILD, DUNETTS_NOTE] = OBSERVED_NOT_POSSIBLE
    data_with_dunetts_observed_second[DUNETTS] = MILD
    data_with_dunetts_observed_third.loc[data_with_dunetts_observed_third[DUNETTS] != SEVERE, DUNETTS_NOTE] = OBSERVED_NOT_POSSIBLE
    data_with_dunetts_observed_third[DUNETTS] = SEVERE

    data_with_dunetts_not_observed_0 = train_data[train_data[DUNETTS] == NOT_OBSERVED].copy()
    data_with_dunetts_not_observed_0[DUNETTS_NOTE] = NOT_OBSERVED_STR
    data_with_dunetts_not_observed_1 = data_with_dunetts_not_observed_0.copy()
    data_with_dunetts_not_observed_2 = data_with_dunetts_not_observed_0.copy()
    data_with_dunetts_not_observed_0[DUNETTS] = ABSENT
    data_with_dunetts_not_observed_1[DUNETTS] = MILD
    data_with_dunetts_not_observed_2[DUNETTS] = SEVERE
    augmented_data = pd.concat(
        [
            data_with_dunetts_observed_first,
            data_with_dunetts_observed_second,
            data_with_dunetts_observed_third,
            data_with_dunetts_not_observed_0,
            data_with_dunetts_not_observed_1,
            data_with_dunetts_not_observed_2
        ],
        ignore_index=True
    )
    return augmented_data


def complete_table(aug_train_data, p_dun, p_fo_dun, p_deg_dun, p_slo_genes_dun, p_genes):
    completed_table = get_prob_dun_given_evidence(aug_train_data, p_dun, p_fo_dun, p_deg_dun, p_slo_genes_dun, p_genes)
    completed_table["denominator"] = completed_table[P_DUNETTS + "0_and_evidence"] + \
                                     completed_table[P_DUNETTS + "1_and_evidence"] + \
                                     completed_table[P_DUNETTS + "2_and_evidence"]
    completed_table[P_DUNETTS + "_and_evidence"] = np.nan
    completed_table.loc[completed_table[DUNETTS] == 0, P_DUNETTS + "_and_evidence"] = completed_table.loc[
        completed_table[DUNETTS] == 0, P_DUNETTS + "0_and_evidence"]
    completed_table.loc[completed_table[DUNETTS] == 1, P_DUNETTS + "_and_evidence"] = completed_table.loc[
        completed_table[DUNETTS] == 1, P_DUNETTS + "1_and_evidence"]
    completed_table.loc[completed_table[DUNETTS] == 2, P_DUNETTS + "_and_evidence"] = completed_table.loc[
        completed_table[DUNETTS] == 2, P_DUNETTS + "2_and_evidence"]
    completed_table[P_DUNETTS_GIVEN_EVIDENCE] = completed_table[P_DUNETTS + "_and_evidence"] / completed_table[
        "denominator"]
    completed_table.loc[completed_table[DUNETTS_NOTE] == OBSERVED_ACTUAL, P_DUNETTS_GIVEN_EVIDENCE] = 1
    completed_table.loc[completed_table[DUNETTS_NOTE] == OBSERVED_ACTUAL, P_DUNETTS + "_and_evidence"] = 1
    completed_table.loc[completed_table[DUNETTS_NOTE] == OBSERVED_NOT_POSSIBLE, P_DUNETTS_GIVEN_EVIDENCE] = 0
    completed_table.loc[completed_table[DUNETTS_NOTE] == OBSERVED_NOT_POSSIBLE, P_DUNETTS + "_and_evidence"] = 0
    return completed_table


def EM(aug_train_data, p_dun, p_fo_dun, p_deg_dun, p_slo_dun_genes, p_genes):
    p_dun = p_dun.copy()
    p_fo_dun = p_fo_dun.copy()
    p_deg_dun = p_deg_dun.copy()
    p_slo_dun_genes = p_slo_dun_genes.copy()
    p_genes = p_genes.copy()
    likelihood_change = 1
    prev_likelihood_indicator = None
    while likelihood_change > 0.01:
        completed_table = complete_table(aug_train_data.copy(), p_dun, p_fo_dun, p_deg_dun, p_slo_dun_genes, p_genes)

        # update P(Dunetts)
        for dunetts_form in [ABSENT, MILD, SEVERE]:
            p_dun.loc[p_dun[DUNETTS] == dunetts_form, PROB] = \
                completed_table[completed_table[DUNETTS] == dunetts_form][P_DUNETTS_GIVEN_EVIDENCE].sum() / \
                completed_table[P_DUNETTS_GIVEN_EVIDENCE].sum()

        # update P(Forennditis | Dunetts) and P(Degar | Dunetts)
        for dunetts_form in [ABSENT, MILD, SEVERE]:
            for symptom in [PRESENT, ABSENT]:
                p_fo_dun.loc[(p_fo_dun[DUNETTS] == dunetts_form) & (p_fo_dun[FORENNDITIS] == symptom), PROB] = \
                    completed_table[(completed_table[DUNETTS] == dunetts_form) &
                                    (completed_table[FORENNDITIS] == symptom)][P_DUNETTS_GIVEN_EVIDENCE].sum() / \
                    completed_table[completed_table[DUNETTS] == dunetts_form][P_DUNETTS_GIVEN_EVIDENCE].sum()

                p_deg_dun.loc[(p_deg_dun[DUNETTS] == dunetts_form) & (p_deg_dun[DEGAR] == symptom), PROB] = \
                    completed_table[(completed_table[DUNETTS] == dunetts_form) &
                                    (completed_table[DEGAR] == symptom)][P_DUNETTS_GIVEN_EVIDENCE].sum() / \
                    completed_table[completed_table[DUNETTS] == dunetts_form][P_DUNETTS_GIVEN_EVIDENCE].sum()

        # update P(sloepnea, tGene | Dunetts)
        for dunetts_form in [ABSENT, MILD, SEVERE]:
            for symptom in [PRESENT, ABSENT]:
                for gene in [PRESENT, ABSENT]:
                    p_slo_dun_genes.loc[
                        (p_slo_dun_genes[DUNETTS] == dunetts_form) &
                        (p_slo_dun_genes[SLOEPNEA] == symptom) &
                        (p_slo_dun_genes[TGENE] == gene),
                        PROB
                    ] = \
                        completed_table[
                            (completed_table[DUNETTS] == dunetts_form) &
                            (completed_table[SLOEPNEA] == symptom) &
                            (completed_table[TGENE] == gene)
                            ][P_DUNETTS_GIVEN_EVIDENCE].sum() / \
                        completed_table[
                            (completed_table[DUNETTS] == dunetts_form) & (completed_table[TGENE] == gene)
                        ][P_DUNETTS_GIVEN_EVIDENCE].sum()

        gene_present = completed_table[TGENE].sum()
        gene_absent = len(completed_table) - gene_present
        p_genes.loc[p_genes[TGENE] == PRESENT, PROB] = gene_present * 1.0 / (gene_present + gene_absent)
        p_genes.loc[p_genes[TGENE] == ABSENT, PROB] = gene_absent * 1.0 / (gene_present + gene_absent)

        if prev_likelihood_indicator is None:
            prev_likelihood_indicator = completed_table[P_DUNETTS + "_and_evidence"].sum()
            continue
        else:
            weight_indicator = completed_table[P_DUNETTS + "_and_evidence"].sum()
            likelihood_change = abs(weight_indicator - prev_likelihood_indicator)
            prev_likelihood_indicator = weight_indicator
    return p_dun, p_fo_dun, p_deg_dun, p_slo_dun_genes, p_genes


def generate_graph(df):
    plt.plot(df["delta"], df["prior_accuracy"], color='r', label="Accuracy before EM")
    plt.plot(df["delta"], df["post_accuracy"], color='g', label="Accuracy after EM")
    plt.xlabel('Delta')
    plt.ylabel('Percentage accuracy (%)')
    plt.title(f"Accuracy")
    plt.legend()
    plt.errorbar(df["delta"], df["prior_accuracy"], yerr=df["prior_stdev"], fmt='o', ecolor='orange', color='black')
    plt.errorbar(df["delta"], df["post_accuracy"], yerr=df["post_stdev"], fmt='o', ecolor='blue', color='black')
    plt.show()


def main():
    train_data = import_data("traindata.txt")
    test_data = import_data("testdata.txt")
    trials = []

    for x in range(10, 400, 20):
        delta = x * 1.0/100
        accuracy_before_EM = []
        accuracy_after_EM = []
        for trial in range(20):
            p_dunnetts, p_forennditis_dunnetts, p_degar_dunnetts, p_sloepnea_dunnetts_genes, p_genes = generate_initial_CPT(delta)
            prior_acc = get_accuracy(test_data, p_dunnetts, p_forennditis_dunnetts, p_degar_dunnetts,
                                     p_sloepnea_dunnetts_genes, p_genes)
            accuracy_before_EM.append(prior_acc)
            augmented_data = augment_data(train_data)
            p_dunnetts, p_forennditis_dunnetts, p_degar_dunnetts, p_sloepnea_dunnetts_genes, p_genes = \
                EM(augmented_data, p_dunnetts, p_forennditis_dunnetts, p_degar_dunnetts, p_sloepnea_dunnetts_genes, p_genes)
            post_acc = get_accuracy(test_data, p_dunnetts, p_forennditis_dunnetts, p_degar_dunnetts,
                                    p_sloepnea_dunnetts_genes, p_genes)
            accuracy_after_EM.append(post_acc)

        trials.append((
            delta,
            statistics.mean(accuracy_before_EM),
            statistics.stdev(accuracy_before_EM),
            statistics.mean(accuracy_after_EM),
            statistics.stdev(accuracy_after_EM)
        ))
    trials_report = pd.DataFrame(
        trials,
        columns=["delta", "prior_accuracy", "prior_stdev", "post_accuracy", "post_stdev"]
    )
    generate_graph(trials_report)


if __name__ == "__main__":
    main()
