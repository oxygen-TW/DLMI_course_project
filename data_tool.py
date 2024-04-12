import os
import random

def load_data(data_root, use="train", data_type="negative"):
    """
    Load data from data_root and return a list of all the files in the specified data path.
    """

    use = use.lower()
    data_type = data_type.lower()

    assert use in ["train", "val", "test"], "use parameter must be \"train\", \"val\", or \"test\""
    assert data_type  in ["positive", "negative"], "use parameter must be \"positive\" or \"negative\""

    #check if data_root exists
    if not os.path.exists(data_root):
        raise FileNotFoundError("Data root does not exist")
    
    data_path = os.path.join(data_root, use, data_type)

    if not os.path.exists(data_path):
        raise FileNotFoundError("Data path does not exist")
    
    #Return the all the files in the data path with data_root
    return [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    
def randomly_load_data(data_root="", split_ratio=(0.7, 0.15, 0.15), random_seed=None):
    """
    Randomly load data from data_root and split it into train, val, and test sets.
    The data is split based on the split_ratio parameter.
    The data is shuffled before splitting.
    The random seed can be set for reproducibility.
    """

    assert len(split_ratio) == 3, "split_ratio must be a tuple of 3 elements"
    assert sum(split_ratio) == 1, "split_ratio must sum to 1"

    random.seed(random_seed)

    use = ["train", "val", "test"]
    data_type = ["positive", "negative"]

    normal_data = []
    covid_data = []

    for u in use:
        for d in data_type:
            data = load_data(data_root, u, d)
            if d == "negative":
                normal_data.extend(data)
            else:
                covid_data.extend(data)

    assert len(normal_data) > 0, "No normal data found"
    assert len(covid_data) > 0, "No covid data found"

    random.shuffle(normal_data)
    random.shuffle(covid_data)
    
    splitted_data = {
            "train": {
                "normal": normal_data[:int(split_ratio[0] * len(normal_data))],
                "covid": covid_data[:int(split_ratio[0] * len(covid_data))]
            },
            "val": {
                "normal": normal_data[int(split_ratio[0] * len(normal_data)):int((split_ratio[0] + split_ratio[1]) * len(normal_data))],
                "covid": covid_data[int(split_ratio[0] * len(covid_data)):int((split_ratio[0] + split_ratio[1]) * len(covid_data))]
            },
            "test": {
                "normal": normal_data[int((split_ratio[0] + split_ratio[1]) * len(normal_data)):],
                "covid": covid_data[int((split_ratio[0] + split_ratio[1]) * len(covid_data)):]
            }
        }
    
    return {
        "info":{
            "train": {
                "normal": len(splitted_data["train"]["normal"]),
                "covid": len(splitted_data["train"]["covid"])
            },
            "val": {
                "normal": len(splitted_data["val"]["normal"]),
                "covid": len(splitted_data["val"]["covid"])
            },
            "test": {
                "normal": len(splitted_data["test"]["normal"]),
                "covid": len(splitted_data["test"]["covid"])
            },
            "total": len(normal_data) + len(covid_data)
        },
        "data": splitted_data
        }



if(__name__ == "__main__"):
    DATA_ROOT = "/Group16T/raw_data/covid_cxr/"
    # print(load_data(data_root=DATA_ROOT, use="train", data_type="NEGATIVE"))
    print(randomly_load_data(data_root="/Group16T/raw_data/covid_cxr/", split_ratio=(0.7, 0.15, 0.15)))
