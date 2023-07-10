# --- imports ---
import csv
import os

from transformers import BertTokenizer, BertModel


def load_data_from_filename(filename):
    print(filename)
    a = [[] for _ in range(10)]
    with open(filename) as file:
        tsv_file = csv.reader(file, delimiter="\t")
        k = 0
        for line in tsv_file:
            if k > 0 and line[1] != line[4] :
                # print(f"-{k} --- {filename} --- {line[1]}/{line[4]} --- : {line}")
                break
            if k <= 7:
                a[k] = line
            k += 1

    # for x in zip(a[0], a[2]):
    #     print(x)

    return a[1]


def import_model(name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(name)
    model = BertModel.from_pretrained(name)
    return tokenizer, model


# # --- get output ---
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)


def main(data_path='./dataset', model_name='bert-base-uncased'):
    # tokenizer, model = import_model(model_name)

    for filename in os.listdir(data_path):
        if filename[-3:] == 'tsv':
            data = load_data_from_filename(os.path.join(data_path, filename))
            # print(data[3])
            # encoding = tokenizer(data[3], padding=True, truncation=True, max_length=128, return_tensors='pt')
            # print('------------------------')
            # print(encoding)
            # print(type(encoding['input_ids']))
            # print(encoding['input_ids'].shape)
            # break
    # save_output(get_output())


if __name__ == '__main__':
    main()