import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CAT_DIR = './shhs1-dataset-0.14.0.csv'
CATEGORIES = ['race', 'gender', 'age_category_s1', 'smokstat_s1', 'bmi']
sub_categories = {
    'race': ['white', 'black', 'other'],
    'gender': ['male', 'female'],
    'age_category_s1': ['35-44', '45-54', '55-64', '65-74', '75-84', '85 or older'],
    'smokstat_s1': ['Never', 'Current', 'Former'],
    'bmi': ['18-20', '20-30', '30-40', '40-50']
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
    no, mild, moderate, severe = [], [], [], []
    for sub_category in df[category].dropna().unique():
        no.append(count_matching_cells(df, category, sub_category, label=0))
        mild.append(count_matching_cells(df, category, sub_category, label=1))
        moderate.append(count_matching_cells(df, category, sub_category, label=2))
        severe.append(count_matching_cells(df, category, sub_category, label=3))
    return no, mild, moderate, severe


def plot_categories():
    df = pd.read_csv(CAT_DIR)
    df['label'] = df.apply(lambda x: ahi_to_label(x[RAW_LABEL_COL]), axis=1)
    df['bmi'] = df.apply(lambda x: 10 * np.round(x['bmi_s1'] / 10), axis=1).dropna()
    for category in CATEGORIES:
        x_axis = np.arange(len(df[category].dropna().unique()))
        no, mild, moderate, severe = get_label_distr(df, category)
        plt.bar(x_axis, no, 0.1, label='Healthy')
        plt.bar(x_axis + .1, mild, 0.1, label='Mild')
        plt.bar(x_axis + .2, moderate, 0.1, label='Moderate')
        plt.bar(x_axis + .3, severe, 0.1, label='Severe')
        plt.xticks(x_axis, sub_categories[category])
        plt.xlabel(category)
        plt.ylabel("Number of Subjects")
        plt.title(f'Condition distribution by {category}')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    plot_categories()