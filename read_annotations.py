import pickle
import gzip

with gzip.open('Dataset/phoenix14t.pami0.train.annotations_only.gzip', 'rb') as f:
    print(f)
    annotations = pickle.load(f)
print(annotations[0])