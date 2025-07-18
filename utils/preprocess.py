'''
数据预处理
'''
import json
import numpy as np

def generate_json(dataset: list, filename: str):
    key = []
    value = []

    for data in dataset:
        idx = data.find(" ")
        word = data[:idx]  # 获取单词部分
        phenome = data[idx+1:]
        if not word.isalpha() or '#' in phenome:  # 如果单词包含非字母字符，则跳过
            continue
        key.append(word)
        value.append(phenome)
    
    json_data = dict(zip(key, value))   
    with open(f"data_{filename}.json", 'w') as f:
        json.dump(json_data, f, indent=2)

    return json_data


trainset_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1


f = open("cmudict.dict",encoding="utf-8")
dataset = f.read().splitlines()
np.random.shuffle(dataset)

n_items = len(dataset)    # 135166

trainset_num = int(trainset_ratio*n_items)  # 94616
valset_num = int(val_ratio*n_items)   # 27033
testset_num = n_items - trainset_num - valset_num # 13517

train_data = dataset[:trainset_num]
valset_data = dataset[trainset_num:trainset_num+valset_num]
test_data = dataset[trainset_num+valset_num:]


train_set = generate_json(train_data,'train')
val_set = generate_json(valset_data,'val')
test_set = generate_json(test_data,'test')


print(len(train_set),len(val_set),len(test_set))
