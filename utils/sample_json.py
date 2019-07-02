import json
import pprint
import numpy as np

pp = pprint.PrettyPrinter(indent=4)

# use json.load() to change the json format string to python object (dict, list...)
# use json.dump() to change the python object (dict, list...) to json format string

with open('data\\skeleton\\run_front_result\\run_front_000000000000_keypoints.json') as json_file:
    data = json.load(json_file)
    pp.pprint(np.array(data['people'][0]['pose_keypoints']).reshape(-1,3))
    pp.pprint(np.array(data['people'][0]['pose_keypoints']).reshape(-1,3).dtype)


json_str = json.dump(data)