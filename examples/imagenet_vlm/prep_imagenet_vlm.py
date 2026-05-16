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
    uv run examples/imagenet_vlm/prep_imagenet_vlm.py \
        --in-dir /Volumes/Lexar/datasets/visual-layer_imagenet-1k-vl-enriched \
        --out-dir /Volumes/Lexar/datasets/visual-layer_imagenet-1k-vl-enriched-dnnc \
        --split train \
        --num-workers 8

Output format (version 2, with caption):
  [64B shard header]     version=2
  [body — tightly packed, NO padding]
    Each sample:
      [int32]  label               (0-999)
      [int32]  text_len            (class_name_bytes + 1 for EOS)
      [int32 × text_len] text_ids  (byte tokens of class name + EOS)
      [int32]  caption_len         (bytes in caption string, 0 = none)
      [uint8 × caption_len] caption_bytes (raw UTF-8 bytes, NO EOS)
      [uint8 × H×W×C]  pixels     (NHWC row-major)
"""
import argparse, json, os, re, struct, gc
from multiprocessing import Pool, cpu_count
import numpy as np
from datasets import load_dataset
from PIL import Image, PngImagePlugin
# Increase max text chunk size to handle large PNG metadata
PngImagePlugin.MAX_TEXT_CHUNK = 1024 * 1024  # 1MB
from tqdm import tqdm
import io

TARGET_H = TARGET_W = 224
EOS_ID = 258
SHARD_VERSION = 2
IDX_ENTRY_BYTES = 20
SHARD_NAME_RE = re.compile(r"^(?P<split>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.(?P<ext>bin|idx)$")

def process_sample(args):
    """Decode JPEG/PNG bytes, resize, crop, pack blob with caption.
    
    Image is raw bytes from parquet.  Worker decodes with PIL inside try/except.
    Returns (raw_pos, blob_bytes, label_id, text_len, caption_len) or
    (raw_pos, None) on error.
    """
    try:
        raw_pos, img_bytes, label_id, label_name, caption_text = args
        
        # Decode from raw bytes inside worker
        img_pil = Image.open(io.BytesIO(img_bytes))
        if img_pil.mode != "RGB":
            img_pil = img_pil.convert("RGB")
        
        # Resize short edge to 256
        w, h = img_pil.size
        s = min(w, h)
        scale = 256.0 / s
        new_w, new_h = int(w * scale), int(h * scale)
        img_pil = img_pil.resize((new_w, new_h), Image.BICUBIC)
        
        # Center crop to 224x224
        left = (new_w - TARGET_W) // 2
        top  = (new_h - TARGET_H) // 2
        img_pil = img_pil.crop((left, top, left + TARGET_W, top + TARGET_H))
        
        # Pixels as NHWC uint8 bytes
        pixels = np.array(img_pil, dtype=np.uint8).tobytes()
        
        # Build class label: byte tokens + EOS
        label_bytes = label_name.encode("ascii")
        text_ids = list(label_bytes) + [EOS_ID]
        text_len = len(text_ids)
        
        # Build caption: raw UTF-8 bytes (no EOS)
        cap_encoded = caption_text.encode("utf-8") if caption_text else b""
        cap_len = len(cap_encoded)
        
        # Pack: label + text_len + text_ids + cap_len + cap_bytes + pixels
        blob = (
            struct.pack("<i", label_id)
            + struct.pack("<i", text_len)
            + struct.pack(f"<{text_len}i", *text_ids)
            + struct.pack("<i", cap_len)
            + cap_encoded
            + pixels
        )
        
        return (raw_pos, blob, label_id, text_len, cap_len)
    except Exception:
        raw_pos = args[0] if args else -1
        return (raw_pos, None)

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

def _parse_shard_name(filename, split_tag, ext):
    m = SHARD_NAME_RE.match(filename)
    if not m or m.group("split") != split_tag or m.group("ext") != ext:
        return None
    return int(m.group("idx")), int(m.group("total"))

def _shard_path(out_dir, split_tag, shard_idx, num_shards, ext):
    return os.path.join(out_dir, f"{split_tag}-{shard_idx+1:05d}-of-{num_shards:05d}.{ext}")

def _idx_entry_count(path):
    return os.path.getsize(path) // IDX_ENTRY_BYTES

def _meta_path(out_dir, split_tag):
    return os.path.join(out_dir, f"{split_tag}.meta.json")

def _read_meta(out_dir, split_tag):
    path = _meta_path(out_dir, split_tag)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def _write_meta(out_dir, split_tag, *, row_limit, raw_seen, samples, skipped,
                samples_per_shard, num_shards, shard_raw_ends):
    path = _meta_path(out_dir, split_tag)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({
            "split": split_tag,
            "row_limit": row_limit,
            "raw_seen": raw_seen,
            "samples": samples,
            "skipped": skipped,
            "samples_per_shard": samples_per_shard,
            "num_shards": num_shards,
            "shard_raw_ends": shard_raw_ends,
        }, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)

def _clean_split_outputs(out_dir, split_tag):
    for filename in os.listdir(out_dir):
        if filename in (f"{split_tag}.idx", f"{split_tag}.meta.json") or _parse_shard_name(filename, split_tag, "bin") or _parse_shard_name(filename, split_tag, "idx"):
            os.remove(os.path.join(out_dir, filename))

def _existing_complete_shards(out_dir, split_tag, samples_per_shard):
    """Return (complete_count, filename_num_shards) and remove first partial shard.

    Resume only from full shards. Final partial shards are recomputed so reruns cannot
    silently append duplicate samples.
    """
    bins = []
    for filename in os.listdir(out_dir):
        parsed = _parse_shard_name(filename, split_tag, "bin")
        if parsed:
            bins.append((parsed[0] - 1, parsed[1], filename))
    if not bins:
        return 0, None

    totals = {total for _, total, _ in bins}
    file_num_shards = max(totals)
    if len(totals) > 1:
        print(f"Warning: mixed shard totals in {out_dir}; resuming files with of-{file_num_shards:05d}")
    bins = sorted((idx, name) for idx, total, name in bins if total == file_num_shards)

    complete = 0
    by_idx = {idx: name for idx, name in bins}
    for idx in range(len(bins)):
        name = by_idx.get(idx)
        if name is None:
            break
        idx_path = _shard_path(out_dir, split_tag, idx, file_num_shards, "idx")
        if not os.path.exists(idx_path):
            break
        n_entries = _idx_entry_count(idx_path)
        if n_entries != samples_per_shard:
            break
        complete += 1

    # Remove first incomplete shard and anything after it for this filename set.
    for idx, name in bins:
        if idx < complete:
            continue
        for ext in ("bin", "idx"):
            path = _shard_path(out_dir, split_tag, idx, file_num_shards, ext)
            if os.path.exists(path):
                os.remove(path)

    return complete, file_num_shards

def _column_pylist(table, name, default=None):
    if name not in table.column_names:
        return default
    return table.column(name).to_pylist()

def _image_bytes_pylist(table):
    col = table.column("image").combine_chunks()
    if hasattr(col, "field"):
        names = getattr(col.type, "names", [])
        if "bytes" in names:
            return col.field("bytes").to_pylist()
    return col.to_pylist()

def _coerce_image_bytes(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("bytes")
    if isinstance(value, memoryview):
        return value.tobytes()
    return value

def _finalize_shard_count(out_dir, split_tag, old_num_shards, actual_num_shards):
    """Rename shard files and patch headers so dataloader sees actual count."""
    if actual_num_shards == old_num_shards:
        return
    for shard_idx in range(actual_num_shards):
        for ext in ("bin", "idx"):
            old = _shard_path(out_dir, split_tag, shard_idx, old_num_shards, ext)
            new = _shard_path(out_dir, split_tag, shard_idx, actual_num_shards, ext)
            if old != new and os.path.exists(old):
                os.replace(old, new)
        bin_path = _shard_path(out_dir, split_tag, shard_idx, actual_num_shards, "bin")
        if os.path.exists(bin_path):
            with open(bin_path, "r+b") as f:
                f.seek(28)  # num_shards field in imagenet_shard_header
                f.write(struct.pack("<I", actual_num_shards))

def main():
    parser = argparse.ArgumentParser(description="Preprocess ImageNet into dnn.c shard format")
    parser.add_argument("--in-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--samples-per-shard", type=int, default=7100)
    parser.add_argument("--limit", type=int, default=0,
                        help="Max samples to process (0 = all)")
    parser.add_argument("--num-workers", type=int, default=cpu_count())
    parser.add_argument("--no-resume", action="store_true",
                        help="Force restart from scratch even if shards exist")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    split_tag = "train" if args.split == "train" else "val"
    
    data_path = os.path.join(args.in_dir, "data")
    all_shards = sorted([
        f for f in os.listdir(data_path)
        if f.startswith(args.split) and f.endswith(".parquet")
    ])
    print(f"Found {len(all_shards)} parquet shards for split '{args.split}'")
    if not all_shards:
        raise FileNotFoundError(f"No parquet shards found in {data_path} for split '{args.split}'")
    if args.samples_per_shard <= 0:
        raise ValueError("--samples-per-shard must be positive")
    if args.num_workers <= 0:
        raise ValueError("--num-workers must be positive")
    
    import pyarrow.parquet as pq
    
    # Get label names and count from first shard
    first_path = os.path.join(data_path, all_shards[0])
    ds = load_dataset("parquet", data_files=first_path, split="train", streaming=False)
    names = ds.features["label"].names
    total_samples = len(ds)
    del ds; gc.collect()
    
    # Count remaining shards
    for shard_file in all_shards[1:]:
        pf = pq.ParquetFile(os.path.join(data_path, shard_file))
        total_samples += pf.metadata.num_rows
    
    if args.limit > 0 and args.limit < total_samples:
        total_samples = args.limit
        print(f"Limiting to {total_samples} samples")
    num_shards = (total_samples + args.samples_per_shard - 1) // args.samples_per_shard
    print(f"Total: {total_samples} samples -> up to {num_shards} output shards")
    
    # Write labels.txt
    labels_path = os.path.join(args.out_dir, "labels.txt")
    with open(labels_path, "w") as f:
        for name in names:
            f.write(name + "\n")
    print(f"Labels: {labels_path} ({len(names)} classes)")
    
    # Validate max label length
    max_stored = max(len(n.encode("ascii")) + 1 for n in names)
    print(f"Max stored text_len: {max_stored} (IMAGENET_MAX_TEXT_LEN >= {max_stored+1})")
    assert max_stored + 1 <= 128, f"Label too long: {max_stored + 1}"
    
    # ── Determine start state: resume or clean ──
    shard_raw_ends = []
    if args.no_resume:
        _clean_split_outputs(args.out_dir, split_tag)
        out_idx, file_num_shards = 0, num_shards
    else:
        out_idx, existing_num_shards = _existing_complete_shards(
            args.out_dir, split_tag, args.samples_per_shard)
        file_num_shards = existing_num_shards or num_shards
        meta = _read_meta(args.out_dir, split_tag)
        if meta and meta.get("samples_per_shard") == args.samples_per_shard:
            shard_raw_ends = list(meta.get("shard_raw_ends", []))[:out_idx]
        if existing_num_shards and existing_num_shards != num_shards:
            print(f"Resume: keeping existing filename count of-{existing_num_shards:05d} (planned {num_shards})")
        if out_idx:
            print(f"Resume: {out_idx} complete shards exist")
    
    blobs, entries, offset = [], [], 0
    resume_rows = shard_raw_ends[-1] if shard_raw_ends else out_idx * args.samples_per_shard
    total_seen = 0
    skipped = 0
    row_limit = args.limit if args.limit > 0 else total_samples
    
    with Pool(args.num_workers) as pool:
        for parquet_file in tqdm(all_shards, desc="Parquet"):
            if total_seen >= row_limit:
                break
            
            parquet_path = os.path.join(data_path, parquet_file)
            pf = pq.ParquetFile(parquet_path)
            n_in_shard = pf.metadata.num_rows
            rows_left = row_limit - total_seen
            rows_this_shard = min(n_in_shard, rows_left)
            
            if total_seen + rows_this_shard <= resume_rows:
                total_seen += rows_this_shard
                continue
            
            # Use pyarrow to read raw columns (avoid HF image decode which crashes on some PNGs).
            schema_names = set(pf.schema_arrow.names)
            columns = ["image", "label"]
            if "caption_enriched" in schema_names:
                columns.append("caption_enriched")
            table = pq.read_table(parquet_path, columns=columns)
            img_col = _image_bytes_pylist(table)
            label_col = table.column("label").to_pylist()
            caption_col = _column_pylist(table, "caption_enriched")
            if caption_col is None:
                caption_col = [""] * len(table)
            del table; gc.collect()
            
            batch_args = []
            for i in range(n_in_shard):
                if total_seen >= row_limit:
                    break
                if total_seen < resume_rows:
                    total_seen += 1
                    continue
                img_raw = _coerce_image_bytes(img_col[i])
                total_seen += 1
                if img_raw is None:
                    skipped += 1
                    continue
                label_id = int(label_col[i])
                caption = caption_col[i]
                if caption is None:
                    caption = ""
                elif not isinstance(caption, str):
                    caption = str(caption)
                batch_args.append((
                    total_seen,  # 1-based raw row position in split
                    img_raw,  # raw JPEG/PNG bytes
                    label_id,
                    names[label_id],
                    caption,
                ))
            
            if not batch_args:
                continue
            
            for result in tqdm(
                pool.imap(process_sample, batch_args),
                total=len(batch_args), desc=f"  {parquet_file}", leave=False,
            ):
                raw_pos = result[0]
                if len(result) == 2 and result[1] is None:
                    skipped += 1
                    continue
                _, blob, lid, tlen, _ = result
                blobs.append(blob)
                entries.append(struct.pack("<IIQHH", out_idx, len(blobs)-1, offset, tlen, lid))
                offset += len(blob)
                
                if len(blobs) >= args.samples_per_shard:
                    _flush(blobs, entries, out_idx, file_num_shards, args.out_dir, split_tag)
                    shard_raw_ends.append(raw_pos)
                    out_idx += 1
                    blobs, entries, offset = [], [], 0
    
    if blobs:
        _flush(blobs, entries, out_idx, file_num_shards, args.out_dir, split_tag)
        shard_raw_ends.append(total_seen)
        out_idx += 1
    
    actual_num_shards = out_idx
    _finalize_shard_count(args.out_dir, split_tag, file_num_shards, actual_num_shards)
    
    # Combined split .idx — concatenate per-shard idx files up to actual_num_shards
    split_idx_path = os.path.join(args.out_dir, f"{split_tag}.idx")
    total_in_idx = 0
    with open(split_idx_path, "wb") as out:
        hdr_pos = out.tell()
        out.write(struct.pack("<IIQ", 0x58444E49, 1, 0))
        for s in range(actual_num_shards):
            p = _shard_path(args.out_dir, split_tag, s, actual_num_shards, "idx")
            if not os.path.exists(p):
                continue
            with open(p, "rb") as f:
                data = f.read()
                total_in_idx += len(data) // IDX_ENTRY_BYTES
                out.write(data)
        out.seek(hdr_pos + 8)
        out.write(struct.pack("<Q", total_in_idx))
    skipped_total = max(0, total_seen - total_in_idx)
    _write_meta(args.out_dir, split_tag,
                row_limit=row_limit,
                raw_seen=total_seen,
                samples=total_in_idx,
                skipped=skipped_total,
                samples_per_shard=args.samples_per_shard,
                num_shards=actual_num_shards,
                shard_raw_ends=shard_raw_ends)
    print(f"  wrote {split_idx_path} ({total_in_idx} samples, {actual_num_shards} shards)")
    print(f"Done. {total_in_idx} samples in {actual_num_shards} shards (planned {num_shards}, skipped {skipped_total}).")

def _flush(blobs, entries, shard_idx, num_shards, out_dir, split_tag):
    bin_path = _shard_path(out_dir, split_tag, shard_idx, num_shards, "bin")
    write_shard(bin_path, blobs, shard_idx, num_shards)
    idx_path = _shard_path(out_dir, split_tag, shard_idx, num_shards, "idx")
    write_idx(idx_path, entries)
    display_total = max(num_shards, shard_idx + 1)
    print(f"  wrote shard {shard_idx+1}/{display_total} ({len(blobs)} samples)")

if __name__ == "__main__":
    main()
