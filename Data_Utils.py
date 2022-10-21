import os
import json
import math
import torch

try:
    import cPickle as _pickle
except ImportError:
    import pickle as _pickle


def serialize(obj, path, in_json=False):
    if in_json:
        with open(path, "w") as file:
            json.dump(obj, file, indent=2)
    else:
        with open(path, 'wb') as file:
            _pickle.dump(obj, file)


def un_serialize(path):
    suffix = os.path.basename(path).split(".")[-1]
    if suffix == "json":
        with open(path, "r") as file:
            return json.load(file)
    else:
        with open(path, 'rb') as file:
            return _pickle.load(file)


def fix_seq_length(seq, max_len):
    seq_ = list(seq)
    if len(seq_) < max_len:
        pad_len = max_len - len(seq_)
        seq_ = seq_ + [0]*pad_len
    elif len(seq_) > max_len:
        seq_ = seq_[-max_len:]
    seq_ = torch.tensor(seq_)
    return seq_


def fix_gps_length(gps_seq, max_len):
    if gps_seq.size(0) < max_len:
        pad_seq = torch.zeros([max_len - gps_seq.size(0), 12], dtype=torch.int)
        gps_seq_ = torch.cat([gps_seq, pad_seq], dim=0)
    elif gps_seq.size(0) > max_len:
        gps_seq_ = gps_seq[-max_len:, :]
    else:
        gps_seq_ = gps_seq
    return gps_seq_


def get_visited_loc(dataset):
    user_visited_loc = {}
    for u in range(len(dataset.user_seq)):
        seq = dataset.user_seq[u]
        user = seq[0][0]
        user_visited_loc[user] = set()
        for i in reversed(range(len(seq))):
            if not seq[i][-1]:
                break
        user_visited_loc[user].add(seq[i][1])
        seq = seq[:i]
        for check_in in seq:
            user_visited_loc[user].add(check_in[1])
    return user_visited_loc


EarthRadius = 6378137
min_lat = -85.05112878
max_lat = 85.05112878
min_lon = -180
max_lon = 180


def clip(n, min_value, max_value):
    return min(max(n, min_value), max_value)


def map_size(level_of_detail):
    return 256 << level_of_detail


def latlon2pxy(lat, lon, level_of_detail):
    lat = clip(lat, min_lat, max_lat)
    lon = clip(lon, min_lon, max_lon)

    x = (lon + 180) / 360
    sin_lat = math.sin(lat * math.pi / 180)
    y = 0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)

    map_size_ = map_size(level_of_detail)
    pixel_x = int(clip(x * map_size_ + 0.5, 0, map_size_ - 1))
    pixel_y = int(clip(y * map_size_ + 0.5, 0, map_size_ - 1))
    return pixel_x, pixel_y


def txy2quadkey(tile_x, tile_y, level_of_detail):
    quadkey = []
    for i in range(level_of_detail, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if (tile_x & mask) != 0:
            digit += 1
        if (tile_y & mask) != 0:
            digit += 2
        quadkey.append(str(digit))

    return ''.join(quadkey)


def pxy2txy(pixel_x, pixel_y):
    tile_x = pixel_x // 256
    tile_y = pixel_y // 256
    return tile_x, tile_y


def latlon2quadkey(lat, lon, level):
    pixel_x, pixel_y = latlon2pxy(lat, lon, level)
    tile_x, tile_y = pxy2txy(pixel_x, pixel_y)
    return txy2quadkey(tile_x, tile_y, level)