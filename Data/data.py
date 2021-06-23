import tensorflow as tf
import random

# x = tf.range(10)
# print(x)

# dataset = tf.data.Dataset.from_tensor_slices(x)
# # print(dataset)

# # for item in dataset:
# #     print(item)

# # dataset2 = tf.data.Dataset.range(10)

# # for item in dataset2:
# #     print(item)

# dataset = dataset.repeat(3).batch(7, drop_remainder= True)

# for item in dataset:
#     print(item)

# dataset = dataset.apply(tf.data.experimental.unbatch())


# for item in dataset:
#     print(item)
# # dataset = tf.data.Dataset.range(x)
# # dataset  = dataset.map(lambda x: x * 2, nums_parallel_calls =True )
# # print([item for item in dataset])


# dataset  = dataset.filter(lambda x: y for y in x if y > 5)

# print(" The new list: ")
# for item in dataset:
#     print(item)

# dataset = tf.data.Dataset.range(10).repeat(3)
# dataset = dataset.shuffle(buffer_size = 5, seed = 42).batch(7, drop_remainder = True)

# for item in dataset:
#     print(item)

train_filepaths = []
test_filepaths = []
validate_filepaths = []


def generate_data():
    csvfile = open('D:\Deep_Learning\Data\datasets\housing\housing.csv', 'r').readlines()
    random.shuffle(csvfile)
    n_rows = len(csvfile)
    #train
    filename = 0
    for i in range(0, int(0.6*n_rows) ):
        if i % 1000 == 0:
            path = 'D:\Deep_Learning\Data\datasets\housing\my_train_' + str(filename) + '.csv'
            open(path, 'w+').writelines(csvfile[i:i+1000])
            filename += 1
            train_filepaths.append(path)
        
    #test
    filename = 0
    for i in range(int(0.6*n_rows), int(0.8*n_rows) ):
        if i % 1000 == 0:
            path = 'D:\Deep_Learning\Data\datasets\housing\my_test_' + str(filename) + '.csv'
            open(path, 'w+').writelines(csvfile[i:i+1000])
            filename += 1
            test_filepaths.append(path)

    #validate
    filename = 0
    for i in range(int(0.8*n_rows), n_rows):
        if i % 1000 == 0:
            path = 'D:\Deep_Learning\Data\datasets\housing\my_validate_' + str(filename) + '.csv'
            open(path, 'w+').writelines(csvfile[i:i+1000])
            filename += 1
            validate_filepaths.append(path)


generate_data()

filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed = 42)

n_readers = 4
dataset = filepath_dataset.interleave(
    lambda filepath:tf.data.TextLineDataset(filepath).skip(1),
    cycle_length = n_readers
)

for line in dataset.take(10):
    print(line.numpy())

X_mean = 0
X_std = 1
n_inputs = 8

def preprocess(line):
    defs = [0.]*n_inputs + [tf.constant([], dtype = tf.float32)]
    fields  = tf.io.decode_csv(line, record_defaults = defs)
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])
    return (x - X_mean) / X_std, y

print(preprocess(b'4.2083,44.0,5.3232,0.9171,846.0,2.3370,37.47,-122.2,2.782'))