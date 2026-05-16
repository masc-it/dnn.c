#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyarrow",
#     "datasets",
#     "pillow",
#     "tqdm",
# ]
# ///
"""Preprocess ImageNet parquet dataset into dnn.c shard format.

Usage:
    uv run scripts/prep_imagenet_vlm.py \
        --in-dir /Volumes/Lexar/datasets/visual-layer_imagenet-1k-vl-enriched \
        --out-dir /Volumes/Lexar/datasets/visual-layer_imagenet-1k-vl-enriched-dnnc \
        --split train \
        --num-workers 8

Output:
    {out_dir}/{split}-{shard:05d}-of-{N:05d}.bin   (raw shard)
    {out_dir}/{split}-{shard:05d}-of-{N:05d}.idx   (per-shard index)
    {out_dir}/{split}.idx                           (combined split index)
    {out_dir}/labels.txt                            (canonical label names)
"""
import argparse, os, struct, sys, gc
from multiprocessing import Pool
import numpy as np
from datasets import load_dataset
from PIL import Image

TARGET_H = TARGET_W = 224
EOS_ID = 258

def worker_init(names_global):
    """Set global names for worker processes."""
    global _names
    _names = names_global

def process_sample(args):
    """Process one sample: decode JPEG, resize, crop, pack blob.
    
    Returns (blob_bytes, label_id, text_len, byte_count).
    """
    img_bytes, label_id, label_name = args
    
    # Decode JPEG
    img = Image.open(img_bytes)
    if img.mode != "RGB":
        img = img.convert("RGB")
    
    # Resize short edge to 256
    w, h = img.size
    s = min(w, h)
    scale = 256.0 / s
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    
    # Center crop to 224x224
    left = (new_w - TARGET_W) // 2
    top  = (new_h - TARGET_H) // 2
    img = img.crop((left, top, left + TARGET_W, top + TARGET_H))
    
    # Pixels as NHWC uint8 bytes
    pixels = np.array(img, dtype=np.uint8).tobytes()
    
    # Build label name -> byte tokens + EOS
    label_str = label_name
    text_ids = list(label_str.encode("ascii")) + [EOS_ID]
    text_len = len(text_ids)
    
    # Pack: label_id(int32) + text_len(int32) + text_ids(int32[]) + pixels(uint8[])
    blob = struct.pack("<i", label_id) + struct.pack("<i", text_len) + struct.pack(f"<{text_len}i", *text_ids) + pixels
    
    return (blob, label_id, text_len, len(pixels))

def write_shard(path, blobs, shard_idx, num_shards):
    """Write a binary shard file."""
    with open(path, "wb") as f:
        hdr = struct.pack("<IIIIIIII", 0x494D474E, 1,
                          TARGET_H, TARGET_W, 3,
                          len(blobs), shard_idx, num_shards)
        f.write(hdr)
        f.write(b'\x00' * 32)
        for blob in blobs:
            f.write(blob)

def write_idx(path, entries):
    """Write a per-shard index file."""
    with open(path, "wb") as f:
        for e in entries:
            f.write(e)

def main():
    parser = argparse.ArgumentParser(description="Preprocess ImageNet into dnn.c shard format")
    parser.add_argument("--in-dir", required=True,
                        help="Path to HF parquet directory")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for shards")
    parser.add_argument("--split", default="train", choices=["train", "validation"],
                        help="Dataset split")
    parser.add_argument("--samples-per-shard", type=int, default=7100,
                        help="Samples per output shard (~1 GB)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Worker processes for parallel JPEG decode")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Map "validation" to the split name used in file names
    split_tag = "train" if args.split == "train" else "val"
    
    # Load dataset (non-streaming for performance, one shard at a time)
    # First get total count and label names
    print("Counting samples and loading label names...")
    data_path = os.path.join(args.in_dir, "data")
    
    # Discover parquet shards
    all_shards = sorted([
        f for f in os.listdir(data_path)
        if f.startswith(args.split) and f.endswith(".parquet")
    ])
    print(f"Found {len(all_shards)} parquet shards for split '{args.split}'")
    
    # Get label names from first shard
    first_path = os.path.join(data_path, all_shards[0])
    ds = load_dataset(
        "parquet",
        data_files=first_path,
        split="train",
        streaming=False,
    )
    names = ds.features["label"].names
    total_samples = len(ds)
    del ds
    gc.collect()
    
    # Count remaining shards quickly (just get row counts)
    for shard_file in all_shards[1:]:
        shard_path = os.path.join(data_path, shard_file)
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(shard_path)
        total_samples += pf.metadata.num_rows
    
    print(f"Total {args.split} samples: {total_samples}")
    
    num_shards = (total_samples + args.samples_per_shard - 1) // args.samples_per_shard
    print(f"Writing {num_shards} shards (~{args.samples_per_shard} samples each)")
    
    # Write labels.txt from raw synset names
    labels_path = os.path.join(args.out_dir, "labels.txt")
    with open(labels_path, "w") as f:
        for name in names:
            f.write(name + "\n")
    print(f"Wrote {labels_path} ({len(names)} labels)")
    
    # Validate label lengths fit within MAX_TEXT_LEN
    max_stored = 0
    for name in names:
        stored = len(name.encode("ascii")) + 1  # +1 for EOS
        if stored > max_stored:
            max_stored = stored
    print(f"Max stored text_len: {max_stored} (need IMAGENET_MAX_TEXT_LEN >= {max_stored + 1})")
    assert max_stored + 1 <= 128, f"Label too long: {max_stored + 1} > 128"
    
    # Process shards
    out_shard_idx = 0
    accumulated_blobs = []
    accumulated_entries = []
    running_offset = 0
    total_processed = 0
    
    from tqdm import tqdm
    
    with Pool(args.num_workers, initializer=worker_init, initargs=(names,)) as pool:
        for parquet_file in tqdm(all_shards, desc="Parquet shards"):
            parquet_path = os.path.join(data_path, parquet_file)
            
            # Load the parquet file in batch
            ds = load_dataset(
                "parquet",
                data_files=parquet_path,
                split="train",
                streaming=False,
            )
            
            # Build args for multiprocessing: (img_bytes_io, label_id, label_name)
            batch_args = []
            for sample in ds:
                img_bytes = sample["image"]["bytes"]
                img_io = __import__("io").BytesIO(img_bytes)
                label_id = sample["label"]
                label_name = names[label_id]
                batch_args.append((img_io, label_id, label_name))
            
            del ds
            gc.collect()
            
            # Process batch in parallel
            results = list(tqdm(
                pool.imap_unordered(process_sample, batch_args),
                total=len(batch_args),
                desc=f"  Shard {parquet_file}",
                leave=False,
            ))
            
            # Sort results to maintain original order (important for idx)
            # Actually, we don't need strict order since we'll shuffle globally later.
            # But the idx must be monotonic within each shard, so let's sort by original position.
            # For now, just use results as they come — we'll shuffle globally in training.
            for blob, lid, tlen, _ in results:
                accumulated_blobs.append(blob)
                accumulated_entries.append(struct.pack(
                    "<IIQHH",
                    out_shard_idx,
                    len(accumulated_blobs) - 1,
                    running_offset,
                    tlen,
                    lid,
                ))
                running_offset += len(blob)
                total_processed += 1
                
                # Flush shard when we have enough
                if len(accumulated_blobs) >= args.samples_per_shard:
                    _flush_shard(accumulated_blobs, accumulated_entries,
                                 out_shard_idx, num_shards, args.out_dir, split_tag)
                    out_shard_idx += 1
                    accumulated_blobs = []
                    accumulated_entries = []
                    running_offset = 0
            
            # Flush any remaining accumulated entries periodically
            if len(accumulated_blobs) >= args.samples_per_shard:
                _flush_shard(accumulated_blobs, accumulated_entries,
                             out_shard_idx, num_shards, args.out_dir, split_tag)
                out_shard_idx += 1
                accumulated_blobs = []
                accumulated_entries = []
                running_offset = 0
    
    # Flush last partial shard
    if accumulated_blobs:
        _flush_shard(accumulated_blobs, accumulated_entries,
                     out_shard_idx, num_shards, args.out_dir, split_tag)
        out_shard_idx += 1
    
    # Write combined split .idx
    _write_split_idx(args.out_dir, split_tag, total_samples, num_shards)
    
    print(f"Done. {total_processed} samples across {num_shards} shards.")

def _flush_shard(blobs, entries, shard_idx, num_shards, out_dir, split_tag):
    """Write one output shard (.bin + .idx)."""
    bin_path = os.path.join(
        out_dir,
        f"{split_tag}-{shard_idx+1:05d}-of-{num_shards:05d}.bin")
    write_shard(bin_path, blobs, shard_idx, num_shards)
    
    idx_path = os.path.join(
        out_dir,
        f"{split_tag}-{shard_idx+1:05d}-of-{num_shards:05d}.idx")
    write_idx(idx_path, entries)
    
    print(f"  wrote shard {shard_idx+1}/{num_shards} ({len(blobs)} samples)")

def _write_split_idx(out_dir, split_tag, total_samples, num_shards):
    """Write combined split .idx (16B header + concatenation)."""
    idx_path = os.path.join(out_dir, f"{split_tag}.idx")
    with open(idx_path, "wb") as out:
        out.write(struct.pack("<IIQ", 0x58444E49, 1, total_samples))
        for s in range(num_shards):
            shard_path = os.path.join(
                out_dir,
                f"{split_tag}-{s+1:05d}-of-{num_shards:05d}.idx")
            with open(shard_path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    out.write(chunk)
    print(f"  wrote {idx_path}")

if __name__ == "__main__":
    main()
