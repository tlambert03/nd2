# ND2 File Format Specification

This document describes the structure of modern ND2 files (versions 2.x and 3.x).
Version 1.x "legacy" files use an entirely different structure and are not covered here.

## Overview

ND2 files are chunked binary files. The file consists of:

1. A **file header chunk** (always first)
2. **Data chunks** (metadata, image data, custom data)
3. A **chunk map** (always last, acts as an index/table of contents)

All multi-byte integers are **little-endian**.

---

## File Structure

### 1. File Header (StartFileChunk)

The file begins with a fixed 112-byte header chunk:

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 4 | uint32 | Magic number: `0x0ABECEDA` |
| 4 | 4 | uint32 | Name length: always `32` |
| 8 | 8 | uint64 | Data length: always `64` |
| 16 | 32 | bytes | Chunk name: `"ND2 FILE SIGNATURE CHUNK NAME01!"` |
| 48 | 64 | bytes | Version string (null-padded), e.g., `"Ver3.0"` |

The version determines the metadata encoding:

- **Version 2.x**: CLX XML encoding (UTF-16 XML wrapped in `<variant>` tags)
- **Version 3.x**: CLX Lite encoding (compact binary format)

### 2. Chunk Structure

Every chunk in the file follows this structure:

```
┌─────────────────────────────────────────────────────────────┐
│ ChunkHeader (16 bytes)                                      │
├────────────┬────────────┬───────────────────────────────────┤
│ magic (4)  │ nameLen(4) │ dataLen (8)                       │
├────────────┴────────────┴───────────────────────────────────┤
│ Chunk Name (nameLen bytes, null-padded to 4K alignment)     │
├─────────────────────────────────────────────────────────────┤
│ Chunk Data (dataLen bytes)                                  │
├─────────────────────────────────────────────────────────────┤
│ Padding (to next 4096-byte boundary)                        │
└─────────────────────────────────────────────────────────────┘
```

**ChunkHeader:**

| Offset | Size | Type | Description |
|--------|------|------|-------------|
| 0 | 4 | uint32 | Magic: `0x0ABECEDA` |
| 4 | 4 | uint32 | Name length (includes padding + null terminator) |
| 8 | 8 | uint64 | Data length |

**Important:** Chunks are aligned to 4096-byte boundaries. The `nameLen` includes extra
padding bytes to ensure the entire chunk (header + name + data) aligns to 4K.

### 3. Chunk Map (File Footer)

The file ends with a chunk map that provides an index to all chunks:

**Footer signature (last 40 bytes of file):**

| Offset from EOF | Size | Type | Description |
|-----------------|------|------|-------------|
| -40 | 32 | bytes | Signature: `"ND2 CHUNK MAP SIGNATURE 0000001!"` |
| -8 | 8 | uint64 | Offset to the chunk map chunk |

**Chunk Map chunk:**

The chunk map itself is a chunk with name `"ND2 FILEMAP SIGNATURE NAME 0001!"`.
Its data contains a sequence of entries:

```
For each chunk:
  ├─ Chunk name (variable length, ends with '!')
  ├─ Offset in file (8 bytes, uint64)
  └─ Size of data (8 bytes, uint64, or same as offset if unknown)
```

The chunk map ends with the signature `"ND2 CHUNK MAP SIGNATURE 0000001!"` followed
by the chunk map offset (repeated for verification).

---

## Chunk Types

### Metadata Chunks

| Chunk Name | Version | Description |
|------------|---------|-------------|
| `ImageAttributesLV!` | 3.x | Image dimensions, bit depth, component count |
| `ImageAttributes!` | 2.x | Same as above, XML format |
| `ImageMetadataLV!` | 3.x | Experiment configuration (loops, channels, etc.) |
| `ImageMetadata!` | 2.x | Same as above, XML format |
| `ImageMetadataSeqLV\|N!` | 3.x | Per-frame metadata (N = frame index) |
| `ImageMetadataSeq\|N!` | 2.x | Same as above, XML format |
| `ImageTextInfoLV!` | 3.x | Human-readable text descriptions |
| `ImageTextInfo!` | 2.x | Same as above, XML format |
| `ImageCalibrationLV\|N!` | 3.x | Calibration data for frame N |

### Image Data Chunks

| Chunk Name | Description |
|------------|-------------|
| `ImageDataSeq\|N!` | Raw image data for frame N |

Image data layout:

- First 8 bytes: timestamp (float64, relative time in milliseconds from acquisition start)
- Remaining bytes: pixel data in row-major order (Y, X, Components)
- Components are interleaved (e.g., RGBRGBRGB... for RGB images)
- May be zlib-compressed if `eCompression` = 0 in attributes

### Custom Data Chunks

Three categories of custom data:

**1. Single-value custom data (`CustomData|name!`):**
Raw binary data, typically arrays of primitive types.

| Common Names | Content |
|--------------|---------|
| `CustomData\|X!`, `Y!`, `Z!` | Stage position arrays (float64 per frame) |
| `CustomData\|PFS_STATUS!`, `PFS_OFFSET!` | Perfect Focus System data |
| `CustomData\|Camera_ExposureTime1!` | Exposure times (float64 per frame) |
| `CustomData\|CameraTemp1!` | Camera temperature readings |
| `CustomData\|AcqTimesCache!` | Acquisition timestamps (float64 per frame) |
| `CustomData\|RoiMetadata_v1!` | ROI definitions |
| `CustomData\|GUIDStore!` | Unique identifiers |

**2. Variable custom data (`CustomDataVar|name!`):**
CLX Lite or CLX XML encoded structured metadata.

| Common Names | Content |
|--------------|---------|
| `CustomDataVar\|AppInfo_V1_0!` | Application information |
| `CustomDataVar\|CustomDataV2_0!` | User-defined experiment metadata |
| `CustomDataVar\|LUTDataV1_0!` | Lookup table / display settings |
| `CustomDataVar\|GrabberCameraSettingsV1_0!` | Camera configuration |

**3. Per-frame sequence data (`CustomDataSeq|name|N!`):**
Data that varies per frame.

| Common Names | Content |
|--------------|---------|
| `CustomDataSeq\|RleZipBinarySequence_N!` | Binary masks (RLE + zlib compressed) |

---

## Data Encoding Formats

### CLX Lite (Version 3.x)

A compact binary format for structured data. Each value is encoded as:

```
┌──────────┬────────────┬────────────────────────┬────────────────┐
│ Type (1) │ NameLen(1) │ Name (NameLen*2 bytes) │ Value (varies) │
└──────────┴────────────┴────────────────────────┴────────────────┘
```

**Type codes:**

| Code | Type | Value Size |
|------|------|------------|
| 1 | bool | 1 byte |
| 2 | int32 | 4 bytes |
| 3 | uint32 | 4 bytes |
| 4 | int64 | 8 bytes |
| 5 | uint64 | 8 bytes |
| 6 | double | 8 bytes |
| 7 | void pointer | 8 bytes |
| 8 | string | UTF-16, null-terminated, variable |
| 9 | byte array | 8-byte length + data |
| 10 | (deprecated) | - |
| 11 | level/struct | 4-byte count + 8-byte length + nested data |
| 76 ('L') | compressed | zlib-compressed CLX Lite data |

**Names:** UTF-16 little-endian, `NameLen` is the number of UTF-16 code units
(actual byte length = NameLen * 2). Names are null-terminated.

**Level (type 11):** Contains nested values. After the header:

- 4 bytes: item count (uint32)
- 8 bytes: total byte length of this level's data (uint64)
- Nested items follow
- After all items: index array (8 bytes per item, offsets)

**Compression (type 76):**

- Skip 10 bytes after type/name header
- Remaining data is zlib-compressed CLX Lite

**Lists:** Represented by items with empty names (`""`). When deserializing,
these become indexed as `i0000000000`, `i0000000001`, etc.

### CLX XML (Version 2.x)

UTF-16 XML with a `<variant>` root element. Values are stored as attributes:

```xml
<?xml version="1.0" encoding="UTF-16"?>
<variant version="1.0">
  <uiWidth runtype="lx_uint32" value="1024"/>
  <dExposureTime runtype="double" value="100.5"/>
  <sDescription runtype="CLxStringW" value="Sample image"/>
  <bEnabled runtype="bool" value="true"/>
  <no_name>
    <item1 runtype="lx_int32" value="1"/>
    <item2 runtype="lx_int32" value="2"/>
  </no_name>
</variant>
```

**Run types (CLxVariant types):**

| runtype | Python equivalent |
|---------|-------------------|
| `lx_int32` | int |
| `lx_uint32` | int |
| `lx_int64` | int |
| `lx_uint64` | int |
| `double` | float |
| `bool` | bool |
| `CLxStringW` | str |
| `CLxByteArray` | bytes |

**Naming convention:** Tag names often have type prefixes:

- `ui` = unsigned int
- `i` = signed int
- `d` = double
- `b` = bool
- `s` / `ws` = string

**Lists:** Represented by `<no_name>` tags or by repeated elements.

---

## Image Data Format

### Attributes (from ImageAttributesLV/ImageAttributes)

The raw CLX Lite data wraps attributes in an `SLxImageAttributes` structure.
Key fields (within `SLxImageAttributes`):

- `uiWidth` / `widthPx`: Width in pixels
- `uiHeight` / `heightPx`: Height in pixels
- `uiComp` / `componentCount`: Components per pixel (channels * RGB components)
- `uiBpcInMemory` / `bitsPerComponentInMemory`: Bits per component (8, 16, or 32)
- `uiBpcSignificant` / `bitsPerComponentSignificant`: Actual significant bits
- `uiSequenceCount` / `sequenceCount`: Total number of frames
- `eCompression`: Compression enum (0=lossless/zlib, 1=lossy, 2=none/uncompressed)

### Pixel Layout

Pixels are stored in row-major order: (Height, Width, Components)

For a 2-channel 16-bit image:

```
Row 0: [Ch0_Px0, Ch1_Px0, Ch0_Px1, Ch1_Px1, ..., Ch0_PxW-1, Ch1_PxW-1]
Row 1: [Ch0_Px0, Ch1_Px0, ...]
...
```

For RGB images: Components are R, G, B interleaved.

### Compression

When `eCompression` = 0 (lossless):

- Image data (after the 8-byte timestamp) is zlib-compressed
- Decompress to get raw pixel data
- Total decompressed size = width × height × components × (bitsPerComponent / 8)

### 16-bit Data Masking

For 16-bit images where `bitsPerComponentSignificant` < 16:

- A mask is applied: `pixel & ((1 << bitsPerComponentSignificant) - 1)`
- This removes high-bit noise from camera ADCs

---

## Reading an ND2 File: Summary

1. **Verify file**: Read first 4 bytes, check for magic `0x0ABECEDA`
2. **Get version**: Parse the header chunk, extract version from data field
3. **Load chunk map**: Seek to EOF-40, read signature and offset, load chunk map
4. **Parse metadata**: Load `ImageAttributesLV!` (or `ImageAttributes!`), decode
5. **Read frames**: For each frame N, load `ImageDataSeq|N!`, skip 8-byte timestamp,
   decompress if needed, reshape to (H, W, C)

---

## Constants

```python
ND2_FILE_SIGNATURE     = b"ND2 FILE SIGNATURE CHUNK NAME01!"  # 32 bytes
ND2_CHUNKMAP_SIGNATURE = b"ND2 CHUNK MAP SIGNATURE 0000001!"  # 32 bytes
ND2_FILEMAP_SIGNATURE  = b"ND2 FILEMAP SIGNATURE NAME 0001!"  # 32 bytes
ND2_CHUNK_MAGIC        = 0x0ABECEDA
CHUNK_ALIGNMENT        = 4096
```

---

## References

- Source: Nikon NIS-Elements SDK (limfile library)
- Key files: `Nd2ChunkedDeviceImpl.cpp`, `CLxVariant.h`, `JsonBridge.cpp`
