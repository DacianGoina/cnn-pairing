"""
IO utils
"""

import json

def export_dict_as_json(dict_obj, filename):
  with open(filename, 'w') as json_file:
    json.dump(dict_obj, json_file, indent=2)

def dict_int64_keys_to_int_keys(dict_obj):
  dict_obj_new = dict()
  for key_val, item_val in dict_obj.items():
    dict_obj_new[int(key_val)] = int(item_val)

  return dict_obj_new

def read_json_file(file_path):

  res = None
  # Open and read the file
  with open(file_path, 'r') as file:
      # Parse the JSON data
      res = json.load(file)

  return res