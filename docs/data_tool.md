
# Documentation for Data Loading and Splitting Functions

## Overview

This script includes two functions, `load_data` and `randomly_load_data`, designed for handling datasets. These functions are particularly useful for machine learning and data analysis tasks, allowing for the loading and random splitting of datasets into training, validation, and testing subsets.

> IMPORTANT: Before using these functions, ensure that the dataset is organized in a specific directory structure. For example, the dataset should have separate folders for training, validation, and testing data, with further subfolders for normal and covid data types.
>
### 1. Function: `load_data`

#### Purpose

Loads data from a given directory and categorizes it according to specified criteria (training, validation, testing set, and data type).

#### Parameters

- `data_root` (string): Root directory for the dataset. Default is an empty string.
- `use` (string): Dataset category (must be "train", "val", or "test"). Default is "train".
- `data_type` (string): Type of data (must be "normal" or "covid"). Default is "normal".

#### Returns

A list of filenames from the specified data path.

#### Exceptions

- Throws `FileNotFoundError` if the data root or specific data path does not exist.
- Validates `use` and `data_type` parameters.

---

### 2. Function: `randomly_load_data`

#### Purpose

Randomly loads and splits data into training, validation, and testing sets based on a given split ratio.

#### Parameters

- `data_root` (string): Root directory for the dataset. Default is an empty string.
- `split_ratio` (tuple): Proportions for splitting data into train, val, and test sets. Must sum to 1. Default is (0.7, 0.15, 0.15).
- `random_seed`: Seed for random number generator. Optional.

#### Returns

A dictionary with two keys:
  - `info`: Contains counts of normal and COVID data in each set (train, val, test).
  - `data`: Actual split data lists for each category (normal, COVID) and set type (train, val, test).

#### Description

- Loads data using `load_data`, separates it into 'normal' and 'COVID', shuffles, and then splits based on `split_ratio`.

---

## Example Usage and Return Examples

### Using `load_data`

```python
data_list = load_data(data_root="/path/to/dataset", use="train", data_type="normal")
```

#### Expected Return

```python
['file1.jpg', 'file2.jpg', ...]  # List of filenames in the train/normal directory.
```

### Using `randomly_load_data`

```python
split_data = randomly_load_data(data_root="/path/to/dataset", split_ratio=(0.7, 0.2, 0.1), random_seed=42)
```

#### Expected Return

```python
{
    "info": {
        "train": {"normal": 700, "covid": 700},
        "val": {"normal": 200, "covid": 200},
        "test": {"normal": 100, "covid": 100},
        "total": 2000
    },
    "data": {
        "train": {"normal": ['train_normal_file1.jpg', ...], "covid": ['train_covid_file1.jpg', ...]},
        "val": {"normal": ['val_normal_file1.jpg', ...], "covid": ['val_covid_file1.jpg', ...]},
        "test": {"normal": ['test_normal_file1.jpg', ...], "covid": ['test_covid_file1.jpg', ...]}
    }
}
```
