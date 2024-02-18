# import libraries
import pickle
import json
import sys
import os
import io


# open pickle file
with open(sys.argv[1], 'rb') as infile:
    obj = pickle.load(infile)

print(obj)

obj = obj.decode().encode('utf-8')

# convert pickle object to json object
json_obj = json.loads(json.dumps(obj, default='utf-8'))


# write the json file
with io.open(
        os.path.splitext(sys.argv[1])[0] + '.json',
        'w',
        encoding='utf8'
    ) as outfile:
    json.dump(json_obj, outfile, ensure_ascii=False, indent=4)


