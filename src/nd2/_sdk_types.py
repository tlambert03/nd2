"""Various raw dict structures likely to be found in an ND2 file."""

from __future__ import annotations

from enum import IntEnum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal, TypedDict, Union

    from typing_extensions import NotRequired, TypeAlias

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

    class RawMetaDict(TypedDict):
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

    class SLxExperimentDict(TypedDict):
        SLxExperiment: RawExperimentDict

    class SubLoopDict(TypedDict):
        uiNextLevelCount: int
        # this is a dict of keys 'i0000000000', 'i0000000001', etc.
        ppNextLevelEx: dict[str, SLxExperimentDict]

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
        # this is a dict of keys 'i0000000000', 'i0000000001', etc.
        pSubLoops: NotRequired[dict[str, SubLoopDict]]

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
        # m_pFilter keys are strings of the form 'i0000000000', 'i0000000001', ...
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

    class RawExperimentRecordDict(TypedDict):
        uiCount: int
        # pEvents keys are strings of the form 'i0000000000', 'i0000000001', ...
        pEvents: dict[str, RawLiteEventDict]

    class RawLiteEventDict(TypedDict, total=False):
        T: float  # timestamp
        T2: NotRequired[float]  # timestamp
        M: int  # EventMeaning
        D: NotRequired[str]  # wsDescription
        A: NotRequired[str]  # wsData
        I: int  # Event ID  # noqa E741
        S: NotRequired[StimulationDict]  # pStimulation

    class StimulationDict(TypedDict):
        T: int  # eType
        L: int  # uiLoopIdx
        P: int  # uiPosition
        D: str  # wsDescription

    class RawTagDict(TypedDict):
        ID: str  # name of the tag
        Type: Literal[
            0,  # Unknown
            1,  # String
            2,  # Int
            3,  # Double
        ]
        Group: Literal[
            0,  # Undefined
            1,  # Device
            2,  # Camera
            3,  # Plugin
            4,  # Macro
        ]
        Size: int
        Desc: str
        Unit: str

    class BinaryMetaDict(TypedDict):
        BinLayerID: int
        State: int
        Color: int
        CompOrder: int
        Name: str
        FileTag: str
        CompName: str
        ColorMode: int

    # These dicts are intermediate dicts created in the process of parsing raw meta
    # they mimic intermediate parsing done by the SDK... but needn't stay this way.

    AxisInterpretation = Literal["distance", "time"]
    CompressionType = Literal["lossless", "lossy", "none"]
    LoopTypeString = Literal[
        "Unknown",
        "TimeLoop",
        "XYPosLoop",
        "XYDiscrLoop",
        "ZStackLoop",
        "PolarLoop",
        "SpectLoop",
        "CustomLoop",
        "NETimeLoop",
        "ManTimeLoop",
        "ZStackLoopAccurate",
    ]

    class ContentsDict(TypedDict):
        frameCount: int

    class GlobalMetadata(TypedDict):
        contents: ContentsDict
        loops: dict[LoopTypeString, int]
        microscope: MicroscopeDict
        position: PositionDict
        time: TimeDict
        volume: VolumeDict

    class MicroscopeDict(TypedDict):
        objectiveMagnification: float | None
        objectiveName: str | None
        objectiveNumericalAperture: float | None
        projectiveMagnification: float | None
        zoomMagnification: float | None
        immersionRefractiveIndex: float | None
        pinholeDiameterUm: float | None

    class PositionDict(TypedDict):
        stagePositionUm: tuple[float, float, float]

    class TimeDict(TypedDict):
        relativeTimeMs: float
        absoluteJulianDayNumber: float

    class VolumeDict(TypedDict):
        axesCalibrated: tuple[bool, bool, bool]
        axesCalibration: tuple[float, float, float]
        axesInterpretation: tuple[
            AxisInterpretation, AxisInterpretation, AxisInterpretation
        ]
        bitsPerComponentInMemory: int
        bitsPerComponentSignificant: int
        cameraTransformationMatrix: tuple[float, float, float, float]
        componentDataType: Literal["float", "unsigned"]
        voxelCount: tuple[int, int, int]


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


class EventMeaning(IntEnum):
    """Meanings of various event types."""

    Unspecified = 0
    Autofocus = 1  # autofocus performed
    UserInteraction1_Old = 2  # the user interactively added the event
    UserInteraction2_Old = 3  # the user interactively added the event
    UserInteraction3_Old = 4  # the user interactively added the event
    UserInteraction4_Old = 5  # the user interactively added the event
    JOBs = 6  # the event added from JOBs
    Command = 7  # a command ran
    Macro = 8  # a macro ran
    Pause = 9  # experiment paused
    Resume = 10  # experiment resumed
    Cancel = 11  # experiment canceled
    RAMGrabZeroTime = 12  # time when RAM capture is triggered (manually, from macro)
    TimeLoopNextPhase = 13  # the time when next phase in time acq was pressed
    Refocus = 14  # experiment paused for refocusing (live started & joysticks enabled)
    Stimulation = 15  # stimulation mark
    ExternalStimulation = 16  # external stimulation mark
    ExperimentStart = 17
    ExperimentEnd = 18
    PhaseXStart = 19
    PhaseXEnd = 20
    BeforeXYMove = 21
    AfterXYMove = 22
    BeforeZSeries = 23
    AfterZSeries = 24
    BeforeLambdaLoop = 25
    AfterLambdaLoop = 26
    BeforeLargeImage = 27
    AfterLargeImage = 28
    BeforeStimulation = 29
    AfterStimulation = 30
    UserEvents = 31
    StreamData = 32
    UserInteraction1 = 33  # the user interactively added the event
    UserInteraction2 = 34  # the user interactively added the event
    UserInteraction3 = 35  # the user interactively added the event
    UserInteraction4 = 36  # the user interactively added the event
    UserInteraction5 = 37  # the user interactively added the event
    UserInteraction6 = 38  # the user interactively added the event
    UserInteraction7 = 39  # the user interactively added the event
    UserInteraction8 = 40  # the user interactively added the event
    BeforeCapture = 41
    AfterCapture = 42
    RealTimeTTLData = 43
    NoAcquisitionStart = 44
    NoAcquisitionEnd = 45
    HardwareError = 46
    StormEvent = 47
    IncubationInfo = 48
    IncubationError = 49
    InteractiveExpEnd = 50
    ExperimentPause = 51
    WIDReplenishmentStart = 52
    WIDReplenishmentEnd = 53
    NSTORM = 54
    EventCount = auto()  # leave this as last

    def description(self) -> str:
        """Return a description of the event meaning."""
        return EVENT_MEANING_DESCRIPTIONS.get(self, "")


EVENT_MEANING_DESCRIPTIONS: dict[EventMeaning, str] = {
    EventMeaning.Unspecified: "Unknown",
    EventMeaning.Autofocus: "Autofocus",
    EventMeaning.UserInteraction1_Old: "User 1 old",
    EventMeaning.UserInteraction2_Old: "User 2 old",
    EventMeaning.UserInteraction3_Old: "User 3 old",
    EventMeaning.UserInteraction4_Old: "User 4 old",
    EventMeaning.JOBs: "JOBs Event",
    EventMeaning.Command: "Command Executed",
    EventMeaning.Macro: "Macro Event",  # retired
    EventMeaning.Pause: "Experiment Paused",
    EventMeaning.Resume: "Experiment Resumed",
    EventMeaning.Cancel: "Experiment Stopped by User",
    EventMeaning.RAMGrabZeroTime: "Acquisition zero time",  # retired
    EventMeaning.TimeLoopNextPhase: "Next Phase Moved by User",
    EventMeaning.Refocus: "Experiment Paused for Refocusing",
    EventMeaning.Stimulation: "Stimulation",
    EventMeaning.ExternalStimulation: "External Stimulation",
    EventMeaning.UserInteraction1: "User 1",
    EventMeaning.UserInteraction2: "User 2",
    EventMeaning.UserInteraction3: "User 3",
    EventMeaning.UserInteraction4: "User 4",
    EventMeaning.UserInteraction5: "User 5",
    EventMeaning.UserInteraction6: "User 6",
    EventMeaning.UserInteraction7: "User 7",
    EventMeaning.UserInteraction8: "User 8",
    EventMeaning.BeforeCapture: "Before Capture",
    EventMeaning.AfterCapture: "After Capture",
    EventMeaning.NoAcquisitionStart: "No Acquisition Phase Start",
    EventMeaning.NoAcquisitionEnd: "No Acquisition Phase End",
    EventMeaning.HardwareError: "Hardware Error",
    EventMeaning.StormEvent: "N-STORM",
    EventMeaning.IncubationInfo: "Incubation Info",
    EventMeaning.IncubationError: "Incubation Error",
    EventMeaning.InteractiveExpEnd: "Interactive Finish",
    EventMeaning.ExperimentPause: "Pause Experiment",
    EventMeaning.WIDReplenishmentStart: "WID Replenishment Start",
    EventMeaning.WIDReplenishmentEnd: "WID Replenishment End",
    EventMeaning.NSTORM: "N-STORM",
}


class StimulationType(IntEnum):
    NoStimulation = 0
    Sequential = 1
    Parallel = 2
    Manual = 3
    Begin = 4
    End = 5
