# plotting utils
import matplotlib.pyplot as plt
import numpy as np
from myutils.process import flatten_list


def normalise_rgb(col_map, alpha=None):
    col_map_norm = {}
    for key, val in col_map.items():
        cols_norm = []
        for v in val:
            cols_norm.append(v / 255)
        if alpha is not None:
            cols_norm.append(alpha)
        col_map_norm[key] = cols_norm
    return col_map_norm


def set_marker_size(x, factor):
    return [x_i**factor for x_i in x]


def grouped_dots():
    # Sample data
    np.random.seed(0)
    num_points = 4
    x = np.array([1, 2, 3])
    categories = ['Category A', 'Category B', 'Category C']
    data = {
        'Category A': {
            'Subcategory 1': np.random.normal(0, 0.1, num_points),
            'Subcategory 2': np.random.normal(-0.1, 0.1, num_points),
            'Subcategory 3': np.random.normal(0.1, 0.1, num_points)
        },
        'Category B': {
            'Subcategory 1': np.random.normal(1, 0.1, num_points),
            'Subcategory 2': np.random.normal(0.9, 0.1, num_points),
            'Subcategory 3': np.random.normal(1.1, 0.1, num_points)
        },
        'Category C': {
            'Subcategory 1': np.random.normal(2, 0.1, num_points),
            'Subcategory 2': np.random.normal(1.9, 0.1, num_points),
            'Subcategory 3': np.random.normal(2.1, 0.1, num_points)
        }
    }

    # Create the grouped dot plot
    fig, ax = plt.subplots()

    subcategories = ['Subcategory 1', 'Subcategory 2', 'Subcategory 3']

    for j, subcategory in enumerate(subcategories):
        y_values = flatten_list([list(data[category][subcategory]) for category in categories])
        x_values = flatten_list([[x[i] + (j - 1) * 0.1] * num_points for i, category in enumerate(categories)])
        ax.plot(x_values, y_values, 'o', label=f'{subcategory}')

    # Customize the plot
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    ax.legend()

    plt.title('Grouped Dot Plot')
    plt.grid(True)

    plt.show()
