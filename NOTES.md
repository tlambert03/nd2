# ND2 notes

## ND2 opening procedure

1. create an IoImageFile instance `f`
    - this calls createIoImageFileDevice, which will make one of the
      following IoImageFileDevice instances, stored as m_fileDevice
        - Nd2FileDevice
        - TifFileDevice
        - JsonFileDevice
2. file is opened with f.open(IoImageFile::OpenMode::ReadOnly).
    - reads version into m_version
    - loads chukmap into m_chunkMap

## Metadata

- rawMetadata -> cachedRawMetadata()
  - checks version, then:

    ```C++
    Bytes imageAttributes = m_chunkedDevice.loadChunk("ImageAttributesLV!");
    Bytes imageExperiment = m_chunkedDevice.loadChunk("ImageMetadataLV!");
    Bytes imageTextInfo = m_chunkedDevice.loadChunk("ImageTextInfoLV!");
    Bytes imageMetadata = m_chunkedDevice.loadChunk(Lim::JsonMetadata::chunkName("ImageMetadataSeqLV|", 0));

    // FOR V3
    m_rawMetadata = Lim::JsonMetadata::readRawMetadataFromLiteVariants(imageAttributes, imageExperiment, imageMetadata, imageTextInfo);
    // OR FOR V2
    m_rawMetadata = Lim::JsonMetadata::readRawMetadataFromVariants(imageAttributes, imageExperiment, imageMetadata, imageTextInfo);
    ```

- attributes -> cachedAttributes
    -> Lim::JsonMetadata::attributes(cachedRawMetadata())
- experiment -> cachedExperiment
    -> Lim::JsonMetadata::experiment(cachedRawMetadata(), cachedAttributes(), m_allLoopIndexes);
- metadata -> cachedMetadata()
    -> Lim::JsonMetadata::metadata(cachedRawMetadata(), cachedGlobalMetadata(), cachedCompRangeMinima(), cachedCompRangeMaxima());
- textInfo -> cachedTextInfo()
    -> Lim::JsonMetadata::textInfo(cachedRawMetadata())
- frameMetadata

    ```C++
        const Doubles& frameTimes = cachedFrameTimes();  // CustomData|AcqTimesCache
        const LoopIndexesVector& allLooopIndexes = cachedAllLoopIndexes();
        return Lim::JsonMetadata::frameMetadata(cachedGlobalMetadata(), cachedMetadata(), cachedExperiment(), frameTimes[seqIndex], allLooopIndexes[seqIndex], metadataPath);
    ```

- cachedGlobalMetadata ->

    ```C++
    m_globalMetadata = Lim::JsonMetadata::globalMetadata(cachedRawMetadata(), cachedExperiment(), cachedTextInfo(), file ? file->fileCreationTime() : 0.0);
    ```

### Things that are cached

- cachedRawMetadata
  - `ImageAttributesLV`
  - `ImageMetadataLV`
  - `ImageTextInfoLV`
  - `ImageMetadataSeqLV|0`
- cachedAttributes (parsed from raw `ImageAttributesLV`)
- cachedExperiment (parsed from raw `ImageMetadataLV` and cachedAttributes)
- cachedGlobalMetadata
- cachedMetadata
- cachedCompRangeMinima / cachedCompRangeMaxima
- cachedTextInfo (parsed from raw `ImageTextInfoLV`)
- cachedFrameTimes (parsed from raw `CustomData|AcqTimesCache`)
- cachedAllLoopIndexes (created during parsing of experiment)
