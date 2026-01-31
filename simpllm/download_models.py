from tqdm import tqdm
import argparse
import base64
import hashlib
import json
import numpy as np
import os
import re
import requests
import sys
import tensorflow as tf

DOWNLOAD_URL_BASE = "https://openaipublic.blob.core.windows.net/gpt-2/models"
MODEL_FILES = [
    "checkpoint",
    "encoder.json",
    "hparams.json",
    "model.ckpt.data-00000-of-00001",
    "model.ckpt.index",
    "model.ckpt.meta",
    "vocab.bpe",
]
SEGMENT_REPACEMENTS = {
    "wte": "tok_embed",
    "wpe": "pos_embed",
    "ln_1": "attn_norm",
    "ln_2": "ffn_norm",
    "ln_f": "final_norm",
    "mlp": "ffn",
    "c_fc": "hidden",
    "c_proj": "output",
    "c_attn": "qkv",
    "b": "bias",
    "g": "scale",
    "w": "weights",
}


def main(size, check_download):
    download_files(size, check_download)

    unpack_dir = f"data/{size}/unpacked"
    os.makedirs(unpack_dir, exist_ok=True)

    meta, reader = model_metadata(size)
    with open(f"{unpack_dir}/metadata.json", "w") as f:
        json.dump(meta, f, sort_keys=True, indent=4)

    # how much to pad the layers counter to
    layers_len = len(str(len("-" * len(meta["layer"]))))

    def fmt_layer_num(n):
        padding = layers_len - len(n)
        return ("0" * padding) + n

    for key in reader.get_variable_to_shape_map():
        nice_key = nice_key_segments(key)
        for i, e in enumerate(nice_key):
            if re.fullmatch("h\\d+", e):
                nice_key[i] = "transformer." + fmt_layer_num(e[1:])
        file_name = ".".join(nice_key)

        tensor = reader.get_tensor(key)
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)

        with open(f"{unpack_dir}/{file_name}.bin", "wb") as f:
            f.write(tensor.tobytes())


def nice_key_segments(k, join_as=None):
    if type(k) is str:
        k = k.split("/")
    if k[0] != "model":
        print(f"unexpected key: {k}")
        exit(1)
    k = k[1:]
    k = [SEGMENT_REPACEMENTS.get(s, s) for s in k]
    return k


def download_files(size, check=False):
    for model_file in MODEL_FILES:
        download_url = f"{DOWNLOAD_URL_BASE}/{size}/{model_file}"
        out_file = f"data/{size}/download/{model_file}"
        download_file(download_url, out_file, check=False)


def download_file(download_url, output_file, check=False):
    print(f"{output_file}: ", end="")
    sys.stdout.flush()
    if os.path.exists(output_file):
        if not check:
            print("file already exists")
            return
        content_md5 = requests.head(download_url).headers.get("Content-MD5")

        if content_md5:
            md5_hash = hashlib.md5()
            with open(output_file, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)

            # Content-MD5 is base64-encoded
            file_md5_b64 = base64.b64encode(md5_hash.digest()).decode()

            if file_md5_b64 == content_md5:
                print("file already exists")
                return
            else:
                print("need to re-download\r", end="")
        else:
            print("\r", end="")

    response = requests.get(download_url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "wb") as f:
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc=output_file
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def model_metadata(size):
    reader = tf.train.load_checkpoint(f"data/{size}/download/model.ckpt")
    # dict. keys are like:
    # - model/h0/mlp/c_proj/b
    # - model/h0/mlp/c_proj/w

    var_to_shape_map = reader.get_variable_to_shape_map()

    metadata = {}
    layers_dict = {}
    for key, val in var_to_shape_map.items():
        key_segments = nice_key_segments(key)
        if re.fullmatch("h\\d+", key_segments[0]):
            key_segments[0] = key_segments[0][1:]  # strip the 'h', and parse to int
            target_dict = layers_dict
        else:
            target_dict = metadata
        for subkey in key_segments[:-1]:
            if subkey not in target_dict:
                target_dict[subkey] = {}
            target_dict = target_dict[subkey]
        if type(val) is list and len(val) == 1:
            val = val[0]
        target_dict[key_segments[-1]] = val

    metadata["layer"] = []

    for layer_key in sorted(layers_dict.keys()):
        metadata["layer"].append(layers_dict[layer_key])

    return metadata, reader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Download models")
    sizes = ["124M", "355M", "774M", "1558M"]
    parser.add_argument("--size", required=False, choices=sizes)
    parser.add_argument("--check-download", action="store_true")
    args = parser.parse_args()
    main(args.size or sizes[0], args.check_download)
