"""Various raw dict structures likely to be found in an ND2 file."""
from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from typing_extensions import Literal, NotRequired, TypeAlias, TypedDict

    class RawAttributesDict(TypedDict, total=False):
        uiWidth: int
        uiWidthBytes: int
        uiHeight: int
        uiComp: int
        uiBpcInMemory: int
        uiBpcSignificant: int
        uiSequenceCount: int
        uiTileWidth: int
        uiTileHeight: int
        eCompression: int
        dCompressionParam: float
        uiVirtualComponents: int

    class RawMetaDict(TypedDict, total=False):
        ePictureXAxis: int
        ePictureYAxis: int
        bCalibrated: bool
        dCalibration: float
        dStgLgCT11: float
        dStgLgCT12: float
        dStgLgCT21: float
        dStgLgCT22: float
        dObjectiveMag: float
        dProjectiveMag: float
        dPinholeRadius: float
        dObjectiveNA: float
        dZoom: float
        dRefractIndex1: float
        dRefractIndex2: float
        wsObjectiveName: str
        dXPos: float
        dYPos: float
        dZPos: float
        dTimeMSec: float
        dTimeAbsolute: float
        sPicturePlanes: PicturePlanesDict

    class RawExperimentDict(TypedDict):
        # these will very likely be present
        aMeasProbesBase64: bytearray
        bControlLight: bool
        bControlShutter: bool
        bUsePFS: bool
        eType: int
        sAutoFocusBeforeLoop: AutoFocusDict
        uLoopPars: LoopParsDict | dict[str, LoopParsDict]
        uiRepeatCount: int
        wsApplicationDesc: str
        wsCommandAfterCapture: str
        wsCommandAfterLoop: str
        wsCommandBeforeCapture: str
        wsCommandBeforeLoop: str
        wsUserDesc: str
        # these may be missing
        bKeepObject: NotRequired[bool]
        bTriggeredStimulation: NotRequired[bool]
        bUseHWSequencer: NotRequired[bool]
        bUseIntenzityCorrection: NotRequired[bool]
        bUseTiRecipe: NotRequired[bool]
        bUseTriggeredAcquisition: NotRequired[bool]
        bUseWatterSupply: NotRequired[bool]
        iRecipeDSCPort: NotRequired[int]
        # either [0,0,1,...] or {'_00': False, '_01': True, '_02': True, ...}
        pItemValid: NotRequired[list[int] | dict[str, bool]]
        pLargeImage: NotRequired[dict]
        pLargeImageEx: NotRequired[dict]
        pNIExperiment: NotRequired[dict]
        # when present this is a dict of keys 'i0000000000', 'i0000000001', etc.
        ppNextLevelEx: NotRequired[dict[str, RawExperimentDict]]
        pRecordedData: NotRequired[dict]
        sParallelExperiment: NotRequired[dict]
        uiNextLevelCount: NotRequired[int]
        vectStimulationConfigurationsSize: NotRequired[int]
        wsCameraName: NotRequired[str]

    class RawTextInfoDict(TypedDict):
        TextInfoItem_0: str
        TextInfoItem_1: str
        TextInfoItem_2: str
        TextInfoItem_3: str
        TextInfoItem_4: str
        TextInfoItem_5: str
        TextInfoItem_6: str
        TextInfoItem_7: str
        TextInfoItem_8: str
        TextInfoItem_9: str
        TextInfoItem_10: str
        TextInfoItem_11: str
        TextInfoItem_12: str
        TextInfoItem_13: str

    class TimeLoopPars(TypedDict):
        bDurationPref: NotRequired[bool]
        dAvgPeriodDiff: float
        dDuration: float
        dMaxPeriodDiff: float
        dMinPeriodDiff: float
        dPeriod: float
        dStart: float
        sAutoFocusBeforeCapture: AutoFocusDict
        uiCount: int
        wsInterfaceName: NotRequired[str]

    # XYPosLoopPars never appears as dict[str, XYPosLoopPars]
    class XYPosLoopPars(TypedDict):
        Points: dict[str, PointDict]
        bKeepPFSOn: bool
        bRedefineAfterAutoFocus: bool
        bRedefineAfterPFS: bool
        bRelativeXY: bool
        bSplitMultipoints: bool
        bUseAFPlane: bool
        bUseZ: bool
        dReferenceX: float
        dReferenceY: float
        sAFBefore: dict
        sZDevice: dict
        uiCount: int

    class ZStackLoopPars(TypedDict):
        bAbsolute: bool
        bTIRF: bool
        bTriggeredPiezo: bool
        bZInverted: bool
        dReferencePosition: float
        dTIRFPFSOffset: float
        dTIRFPosition: float
        dZHigh: float
        dZHome: float
        dZLow: float
        dZStep: float
        iType: int
        uiCount: int
        wsCommandAfterCapture: str
        wsCommandBeforeCapture: str
        wsZDevice: str

    class SpectLoopPars(TypedDict):
        Points: dict[str, SpectLoopPointDict]
        bAskForFilter: NotRequired[bool]  # second to go
        bMergeCameras: bool
        bWaitForPFS: NotRequired[bool]  # first to go
        iOffsetReference: NotRequired[int]  # second to go
        pPlanes: PicturePlanesDict
        uiCount: NotRequired[int]

    class NETimeLoopPars(TypedDict):
        # keys are '_00' or 'i0000000000' ...
        pPeriod: dict[str, PeriodDict]
        pPeriodValid: list[int] | dict[str, bool]
        sAutoFocusBeforeCapture: AutoFocusDict
        sAutoFocusBeforePeriod: AutoFocusDict
        uiCount: int
        uiPeriodCount: int
        wsCommandAfterPeriod: str
        wsCommandBeforePeriod: str

    LoopParsDict: TypeAlias = Union[
        TimeLoopPars, XYPosLoopPars, ZStackLoopPars, SpectLoopPars, NETimeLoopPars
    ]

    class PointDict(TypedDict):
        dPFSOffset: float
        dPosX: float
        dPosY: float
        dPosZ: float
        dPosName: str
        pPosName: NotRequired[str]  # never seen it, but the SDK looks for it.

    class SpectLoopPointDict(TypedDict):
        pAutoFocus: AutoFocusDict
        pZStackPos: int
        pdOffset: NotRequired[float]
        wsCommandAfterCapture: str
        wsCommandBeforeCapture: str

    class AutoFocusDict(TypedDict, total=False):
        eType: int
        dStep: float
        dRange: float
        dPrecision: float
        dSpeed: float
        dOffset: float
        aFocusPlane: dict
        uiFlags: int
        iFocusCriterion: int
        sZDrive: str
        bPiezoZTriggered: bool
        wszOptConf: str
        wszPreferedAFChannel: str
        dCoeff_0: float
        dCoeff_1: float
        dCoeff_2: float
        dCoeff_3: float

    class PeriodDict(TimeLoopPars):
        dIncubationDuration: NotRequired[float]
        sAutoFocusBeforePeriod: AutoFocusDict
        uiGroup: int
        uiLoopType: int

    class PicturePlanesDict(TypedDict, total=False):
        eRepresentation: int
        sDescription: str
        # only one of these two Plane keys will likely be present
        sPlane: dict[str, PlaneDict]
        sPlaneNew: dict[str, PlaneDict]
        # keys are 'a0', 'a1', 'a2', etc.
        sSampleSetting: dict[str, SampleSettingDict]
        uiCompCount: int
        uiCount: int
        uiSampleCount: int

    class PlaneDict(TypedDict, total=False):
        uiCompCount: int
        uiSampleIndex: int
        uiModalityMask: int
        pFluorescentProbe: FluorescentProbeDict
        pFilterPath: FilterPathDict
        dLampVoltage: float
        dFadingCorr: float
        uiColor: int
        sDescription: str
        dAcqTime: float
        dPinholeDiameter: float
        iChannelSeriesIndex: int
        dObjCalibration1to1: float
        eModality: int
        # 'sizeObjFullChip.cx': int
        # 'sizeObjFullChip.cy': int

    class FilterPathDict(TypedDict, total=False):
        m_sDescr: str
        m_uiCount: int
        # m_pFilter keys are strings of the form 'i0000000000
        # i0000000001', ...
        m_pFilter: dict[str, FilterDict]

    class FilterDict(TypedDict, total=False):
        m_sName: str
        m_ePlacement: int
        m_eNature: int
        m_eSpctType: int
        m_uiColor: int
        m_ExcitationSpectrum: SpectrumDict
        m_EmissionSpectrum: SpectrumDict
        m_MirrorSpectrum: SpectrumDict

    class FluorescentProbeDict(TypedDict, total=False):
        m_sName: str
        m_uiColor: int
        m_ExcitationSpectrum: SpectrumDict
        m_EmissionSpectrum: SpectrumDict

    class SpectrumDict(TypedDict, total=False):
        uiCount: int
        # pPoint keys are likely strings of the form 'Point0
        # Point1', ...
        pPoint: dict[str, SpectrumPointDict]
        bPoints: bool

    class SpectrumPointDict(TypedDict, total=False):
        eType: int
        #   eSptInvalid = 0,
        #   eSptPoint = 1,
        #   eSptRaisingEdge = 2,
        #   eSptFallingEdge = 3,
        #   eSptPeak = 4,
        #   eSptRange = 5
        dWavelength: float  # this is usually the one with the value
        uiWavelength: int
        dTValue: float

    class SampleSettingDict(TypedDict, total=False):
        baScanArea: bytearray
        dExposureTime: float
        dObjectiveToPinholeZoom: float
        dRelayLensZoom: float
        dScalingToIntensity: float
        matCameraToStage: dict
        pCameraSetting: dict
        pDeviceSetting: dict  # tons of keys in here
        pObjectiveSetting: dict
        sOpticalConfigs: dict[str, dict]
        sSpecSettings: str
        uiModeFQ: int
        uiOpticalConfigs: int

    class MatrixDict(TypedDict):
        Columns: int
        Rows: int
        Data: list[int]

    # These dicts are intermediate dicts created in the process of parsing raw meta
    # they mimic intermediate parsing done by the SDK... but needn't stay this way.

    CompressionType = Literal["lossless", "lossy", "none"]

    class ContentsDict(TypedDict):
        frameCount: int

    class GlobalMetadata(TypedDict):
        contents: ContentsDict
        loops: dict
        microscope: dict
        position: dict
        time: dict
        volume: dict


class ELxModalityMask(IntEnum):
    fluorescence = 0x0000000000000001
    brightfield = 0x0000000000000002
    phaseContrast = 0x0000000000000010
    diContrast = 0x0000000000000020
    camera = 0x0000000000000100
    laserScanConfocal = 0x0000000000000200
    spinningDiskConfocal = 0x0000000000000400
    sweptFieldConfocalSlit = 0x0000000000000800
    sweptFieldConfocalPinhole = 0x0000000000001000
    dsdConfocal = 0x0000000000002000
    SIM = 0x0000000000004000
    iSIM = 0x0000000000008000
    RCM = 0x0000000000000040  # point scan detected by camera (multiple detection)
    VCS = 0x0000000000000080  # VideoConfocal super-resolution
    sora = 0x0000000040000000  # Yokogawa in Super-resolution mode
    liveSR = 0x0000000000040000
    multiphoton = 0x0000000000010000
    TIRF = 0x0000000000020000
    pmt = 0x0000000000100000
    spectral = 0x0000000000200000
    vaasIF = 0x0000000000400000
    vaasNF = 0x0000000000800000
    transmitDetector = 0x0000000001000000
    nonDescannedDetector = 0x0000000002000000
    virtualFilter = 0x0000000004000000
    gaasp = 0x0000000008000000
    remainder = 0x0000000010000000
    aux = 0x0000000020000000

    @staticmethod
    def get(key: int, default: int) -> int:
        return _MODALITY_MASK_MAP.get(key, default)

    @staticmethod
    def flags(modality_mask: int, component_count: int) -> list[str]:
        if not ELxModalityMask.is_valid(modality_mask):
            return ["brightfield"] if component_count == 3 else ["fluorescence"]
        return [e.name for e in ELxModalityMask if e & modality_mask]

    @staticmethod
    def is_valid(mask: int) -> bool:
        # sourcery skip: remove-unnecessary-cast
        return bool(mask & MaskCombo.LX_ModMaskLight != 0)


class MaskCombo(IntEnum):
    EXCLUDE_LIGHT = 0x00000000000FF000
    LX_ModMaskLight = ELxModalityMask.fluorescence | ELxModalityMask.brightfield
    LX_ModMaskContrast = ELxModalityMask.phaseContrast | ELxModalityMask.diContrast
    LX_ModMaskAcqHWType = (
        ELxModalityMask.camera
        | ELxModalityMask.laserScanConfocal
        | ELxModalityMask.spinningDiskConfocal
        | ELxModalityMask.sweptFieldConfocalSlit
        | ELxModalityMask.sweptFieldConfocalPinhole
        | ELxModalityMask.dsdConfocal
        | ELxModalityMask.RCM
        | ELxModalityMask.VCS
        | ELxModalityMask.iSIM
    )
    LX_ModMaskDetector = (
        ELxModalityMask.spectral
        | ELxModalityMask.vaasIF
        | ELxModalityMask.vaasNF
        | ELxModalityMask.transmitDetector
        | ELxModalityMask.nonDescannedDetector
        | ELxModalityMask.virtualFilter
        | ELxModalityMask.aux
    )


class ELxModality(IntEnum):
    eModWidefieldFluo = 0
    eModBrightfield = 1
    eModLaserScanConfocal = 2
    eModSpinDiskConfocal = 3
    eModSweptFieldConfocal = 4
    eModMultiPhotonFluo = 5
    eModPhaseContrast = 6
    eModDIContrast = 7
    eModSpectralConfocal = 8
    eModVAASConfocal = 9
    eModVAASConfocalIF = 10
    eModVAASConfocalNF = 11
    eModDSDConfocal = 12
    eModMaxValue = 12


_MODALITY_MASK_MAP: dict[int, int] = {
    ELxModality.eModWidefieldFluo: (
        ELxModalityMask.fluorescence | ELxModalityMask.camera
    ),
    ELxModality.eModBrightfield: ELxModalityMask.brightfield | ELxModalityMask.camera,
    ELxModality.eModLaserScanConfocal: (
        ELxModalityMask.fluorescence | ELxModalityMask.laserScanConfocal
    ),
    ELxModality.eModSpinDiskConfocal: (
        ELxModalityMask.fluorescence | ELxModalityMask.spinningDiskConfocal
    ),
    ELxModality.eModSweptFieldConfocal: (
        ELxModalityMask.fluorescence | ELxModalityMask.sweptFieldConfocalSlit
    ),
    ELxModality.eModMultiPhotonFluo: (
        ELxModalityMask.fluorescence
        | ELxModalityMask.multiphoton
        | ELxModalityMask.laserScanConfocal
    ),
    ELxModality.eModPhaseContrast: (
        ELxModalityMask.brightfield | ELxModalityMask.phaseContrast
    ),
    ELxModality.eModDIContrast: (
        ELxModalityMask.brightfield | ELxModalityMask.diContrast
    ),
    ELxModality.eModSpectralConfocal: (
        ELxModalityMask.fluorescence
        | ELxModalityMask.spectral
        | ELxModalityMask.laserScanConfocal
    ),
    ELxModality.eModVAASConfocal: (
        ELxModalityMask.fluorescence
        | ELxModalityMask.vaasNF
        | ELxModalityMask.laserScanConfocal
    ),
    ELxModality.eModVAASConfocalIF: (
        ELxModalityMask.fluorescence
        | ELxModalityMask.vaasIF
        | ELxModalityMask.laserScanConfocal
    ),
    ELxModality.eModVAASConfocalNF: (
        ELxModalityMask.fluorescence
        | ELxModalityMask.vaasNF
        | ELxModalityMask.laserScanConfocal
    ),
    ELxModality.eModDSDConfocal: (ELxModalityMask.dsdConfocal),
}
