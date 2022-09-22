# Copyright 2022 Twitter, Inc.
# SPDX-License-Identifier: Apache-2.0


import atexit
import json
import os
import os.path as osp
import time

class LOG_DATA:
    save_it = 0
    headers = []
    current_row_data = {}
    base_output_dir = None
    output_file_name = None
    output_weights = None
    first_row = True


def configure_output_dir(d=None, force=False):
    LOG_DATA.base_output_dir = d or "experiments_data/temp/{}".format(int(time.time()))
    if not force:
        assert not osp.exists(LOG_DATA.base_output_dir)
    LOG_DATA.output_weights = "{}/weights".format(LOG_DATA.base_output_dir)
    os.makedirs(LOG_DATA.output_weights)
    LOG_DATA.output_file_name = open(osp.join(LOG_DATA.base_output_dir, "log.txt"), 'w')
    # registering a function to be executed at termination
    atexit.register(LOG_DATA.output_file_name.close)
    LOG_DATA.first_row = True
    LOG_DATA.save_it = 0
    LOG_DATA.headers.clear()
    LOG_DATA.current_row_data.clear()
    print("Logging data to directory {}".format(LOG_DATA.output_file_name.name))


def save_params(params):
    with open(osp.join(LOG_DATA.base_output_dir, 'params.json'), 'w') as out:
        out.write(json.dumps(params, indent=2, separators=(',', ': ')))


def load_params(dir):
    with open(osp.join(dir, "params.json"), 'r') as inp:
        data = json.loads(inp.read())
    return data

def log_key_val(key, value):
    assert key not in LOG_DATA.current_row_data, "key already recorded {}".format(key)
    if LOG_DATA.first_row:
        LOG_DATA.headers.append(key)
    else:
        assert key in LOG_DATA.headers, "key not present in headers: {}".format(key)
    LOG_DATA.current_row_data[key] = value

def log_iteration():
    vals = []
    key_lens = [len(key) for key in LOG_DATA.headers]
    max_key_len = max(15, max(key_lens))
    keystr = '%' + '%d' % max_key_len
    fmt = "| " + keystr + "s = %15s |"
    n_slashes = 22 + max_key_len
    print("+" * n_slashes)
    for key in LOG_DATA.headers:
        val = LOG_DATA.current_row_data.get(key, "")
        if hasattr(val, "__float__"):
            valstr = "%8.3g" % val
        else:
            valstr = val
        print(fmt % (key, valstr))
        vals.append(val)
    print("+" * n_slashes)
    if LOG_DATA.output_file_name is not None:
        if LOG_DATA.first_row:
            LOG_DATA.output_file_name.write("\t".join(LOG_DATA.headers))
            LOG_DATA.output_file_name.write("\n")
            LOG_DATA.first_row = False
        LOG_DATA.output_file_name.write("\t".join(map(str, vals)))
        LOG_DATA.output_file_name.write("\n")
        LOG_DATA.output_file_name.flush()
    LOG_DATA.current_row_data.clear()


