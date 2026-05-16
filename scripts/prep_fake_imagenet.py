#!/usr/bin/env python3
"""Generate fake ImageNet shards for testing the VLM dataloader.

No numpy dependency.  Uses random + struct only.

Usage: python scripts/prep_fake_imagenet.py --data-dir data/imagenet --num-samples 500 --num-shards 3
"""

import argparse, os, struct, sys, random

TARGET_H = TARGET_W = 224
EOS_ID = 258
SHARD_VERSION = 2
IMAGENET_CLASSES = 1000

# Fake class names of varying lengths
random.seed(42)
CLASS_NAMES = []
for i in range(IMAGENET_CLASSES):
    if i < 100:
        name = f"c{i:03d}"                           # 5 bytes
    elif i < 300:
        name = f"class_{i:03d}_variety"              # ~18 bytes
    elif i < 600:
        name = f"label_{i:03d}_of_imagenet_set"      # ~27 bytes
    else:
        name = f"cls{i:03d}_short"                   # 12 bytes, always fit
    # Ensure stored_len + 1 <= 31 (bucket fits)
    name_bytes = name.encode("ascii")
    if len(name_bytes) > 30:
        name_bytes = name_bytes[:30]
    CLASS_NAMES.append(name_bytes.decode("ascii"))

def encode_label(label_str: str) -> bytes:
    ids = list(label_str.encode("ascii")) + [EOS_ID]
    return struct.pack("<i", len(ids)) + struct.pack(f"<{len(ids)}i", *ids)

def random_pixels() -> bytes:
    """Generate H*W*C random uint8 bytes."""
    return bytes(random.randint(0, 255) for _ in range(TARGET_H * TARGET_W * 3))

def make_sample(label_id: int, caption: str = "") -> bytes:
    label_str = CLASS_NAMES[label_id]
    # Class label: byte tokens + EOS
    class_ids = list(label_str.encode("ascii")) + [EOS_ID]
    text_len = len(class_ids)
    # Caption: raw bytes, no EOS
    cap_encoded = caption.encode("utf-8")
    pixels = random_pixels()
    return (struct.pack("<i", label_id)
            + struct.pack("<i", text_len)
            + struct.pack(f"<{text_len}i", *class_ids)
            + struct.pack("<i", len(cap_encoded))
            + cap_encoded
            + pixels)

def write_shard(path, blobs, shard_idx, num_shards):
    with open(path, "wb") as f:
        hdr = struct.pack("<IIIIIIII", 0x494D474E, SHARD_VERSION,
                          TARGET_H, TARGET_W, 3,
                          len(blobs), shard_idx, num_shards)
        f.write(hdr)
        f.write(b'\x00' * 32)
        for blob in blobs:
            f.write(blob)

def write_idx(path, entries):
    with open(path, "wb") as f:
        for e in entries:
            f.write(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/imagenet")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # Write labels.txt
    labels_path = os.path.join(args.data_dir, "labels.txt")
    with open(labels_path, "w") as f:
        for name in CLASS_NAMES:
            f.write(name + "\n")
    print(f"Wrote {labels_path} ({len(CLASS_NAMES)} labels)")

    total = args.num_samples
    num_shards = args.num_shards
    samples_per_shard = (total + num_shards - 1) // num_shards
    remaining = total

    for shard_idx_out in range(num_shards):
        n = samples_per_shard if remaining >= samples_per_shard else remaining
        if n <= 0:
            break

        blobs = []
        idx_entries = []
        running_offset = 0

        for local_idx in range(n):
            lbl = random.randint(0, IMAGENET_CLASSES - 1)
            blob = make_sample(lbl)
            blobs.append(blob)

            text_len = struct.unpack_from("<i", blob, 4)[0]
            idx_entries.append(struct.pack(
                "<IIQHH", shard_idx_out, local_idx,
                running_offset, text_len, lbl))
            running_offset += len(blob)

        bin_path = os.path.join(
            args.data_dir,
            f"{args.split}-{shard_idx_out+1:05d}-of-{num_shards:05d}.bin")
        write_shard(bin_path, blobs, shard_idx_out, num_shards)

        idx_path = os.path.join(
            args.data_dir,
            f"{args.split}-{shard_idx_out+1:05d}-of-{num_shards:05d}.idx")
        write_idx(idx_path, idx_entries)

        print(f"  wrote shard {shard_idx_out+1}/{num_shards} ({n} samples)")
        remaining -= n

    # Combined split .idx
    split_idx_path = os.path.join(args.data_dir, f"{args.split}.idx")
    with open(split_idx_path, "wb") as out:
        out.write(struct.pack("<IIQ", 0x58444E49, 1, total))
        for s in range(num_shards):
            idx_path = os.path.join(
                args.data_dir,
                f"{args.split}-{s+1:05d}-of-{num_shards:05d}.idx")
            with open(idx_path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    out.write(chunk)
    print(f"  wrote {split_idx_path}")
    print(f"Done. {total} samples across {num_shards} shards.")

if __name__ == "__main__":
    main()
