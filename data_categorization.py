import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CAT_DIR = './shhs1-dataset-0.14.0.csv'
CATEGORIES = ['race', 'gender', 'age_category_s1', 'smokstat_s1', 'bmi']
sub_categories = {
    'race': ['white', 'black', 'other'],
    'gender': ['male', 'female'],
    'age_category_s1': ['35-44', '45-54', '55-64', '65-74', '75-84', '85+'],
    'smokstat_s1': ['Never', 'Current', 'Former'],
    'bmi': ['18-22', '22-26', '26-30', '30-34', '34-41', '41-50']
}
RAW_LABEL_COL = 'ahi_a0h3a'


def count_occurances(df, cat_col_name):
    occ = dict()
    val_list = list(df[cat_col_name])
    for val in val_list:
        occ[val] = occ.setdefault(val, 0) + 1
    return occ


def ahi_to_label(ahi):
    if ahi < 5:
        return 0
    elif ahi < 15:
        return 1
    elif ahi < 30:
        return 2
    else:
        return 3


def count_matching_cells(df, category, sub_category, label):
    cat_df = df[df['label'] == label]
    cat_df = cat_df[cat_df[category] == sub_category]
    return len(cat_df)


def get_label_distr(df, category):
    no, mild, moderate, severe, totals = [], [], [], [], []
    for sub_category in df[category].dropna().unique():
        no_count = count_matching_cells(df, category, sub_category, label=0)
        mild_count = count_matching_cells(df, category, sub_category, label=1)
        moderate_count = count_matching_cells(df, category, sub_category, label=2)
        severe_count = count_matching_cells(df, category, sub_category, label=3)

        total = no_count + mild_count + moderate_count + severe_count

        no.append(no_count / total)
        mild.append(mild_count / total)
        moderate.append(moderate_count / total)
        severe.append(severe_count / total)
        totals.append(total)
    return no, mild, moderate, severe, totals


def add_total_to_x_axis(category, totals):
    curr_sub_categories = sub_categories[category]
    sub_categories_with_totals = list()
    for sub_cat, total in zip(curr_sub_categories, totals):
        sub_categories_with_totals.append(sub_cat + f'({total})')
    sub_categories[category] = sub_categories_with_totals

def quantize_bmi(bmi):
    if bmi < 22:
        return 18
    elif 22 < bmi < 26:
        return 22
    elif 22 < bmi < 26:
        return 22
    elif 26 < bmi < 30:
        return 26
    elif 30 < bmi < 34:
        return 30
    elif 34 < bmi < 41:
        return 34
    elif 41 < bmi:
        return 41

def plot_categories():
    df = pd.read_csv(CAT_DIR)
    df['label'] = df.apply(lambda x: ahi_to_label(x[RAW_LABEL_COL]), axis=1)
    df['bmi'] = df.apply(lambda x: quantize_bmi(x['bmi_s1']), axis=1).dropna()
    for category in CATEGORIES:
        x_axis = np.arange(len(df[category].dropna().unique()))
        no, mild, moderate, severe, totals = get_label_distr(df, category)
        add_total_to_x_axis(category, totals)
        plt.bar(x_axis, no, 0.1, label='Healthy')
        plt.bar(x_axis + .1, mild, 0.1, label='Mild')
        plt.bar(x_axis + .2, moderate, 0.1, label='Moderate')
        plt.bar(x_axis + .3, severe, 0.1, label='Severe')
        plt.xticks(x_axis, sub_categories[category])
        plt.xlabel(category)
        plt.ylabel("% of subjects")
        plt.title(f'Condition distribution by {category}')
        plt.legend()
        plt.show()

def plot_label_distribution():
    df = pd.read_csv(CAT_DIR)
    labels = df.apply(lambda x: ahi_to_label(x[RAW_LABEL_COL]), axis=1)
    label_counts = labels.value_counts(normalize=True)

    x_axis = ['Healthy', 'Mild', 'Moderate', 'Severe']
    plt.bar(x_axis, label_counts, 0.1)
    plt.xlabel('OSA severity')
    plt.ylabel("% of subjects")
    plt.title(f'OSA severity distribution (Total: {len(df)})')
    plt.show()


if __name__ == '__main__':
    # plot_categories()
    plot_label_distribution()