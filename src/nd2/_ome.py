from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from nd2 import __version__

from ._util import AXIS

try:
    import ome_types.model as m
except ImportError:
    raise ImportError("ome-types is required to read OME metadata") from None

import ome_types.model.channel as channel
from ome_types.model.channel import AcquisitionMode, ContrastMethod, IlluminationType
from ome_types.model.pixels import DimensionOrder
from ome_types.model.simple_types import UnitsLength, UnitsTime

if TYPE_CHECKING:
    from nd2 import ND2File

    from ._sdk_types import RawMetaDict
    from .readers import ModernReader
    from .structures import ModalityFlags


def nd2_ome_metadata(f: ND2File) -> m.OME:
    if f.is_legacy:
        raise NotImplementedError("OME metadata is not available for legacy files")
    rdr = cast("ModernReader", f._rdr)
    meta = f.metadata

    ch0 = next(iter(meta.channels or ()), None)
    channels = [
        m.Channel(
            id=f"Channel:{c_idx}",
            name=ch.channel.name,
            acquisition_mode=ome_acquisition_mode(ch.microscope.modalityFlags),
            color=ch.channel.colorRGB,
            contrast_method=ome_contrast_method(ch.microscope.modalityFlags),
            pinhole_size=ch.microscope.pinholeDiameterUm,
            pinhole_size_unit=UnitsLength.MICROMETER,  # default is "um"
            emission_wavelength=ch.channel.emissionLambdaNm,
            emission_wavelength_unit=UnitsLength.NANOMETER,  # default is "nm"
            excitation_wavelength=ch.channel.excitationLambdaNm,
            excitation_wavelength_unit=UnitsLength.NANOMETER,  # default is "nm"
            illumination_type=ome_illumination_type(ch.microscope.modalityFlags),
            # detector_settings=...,
            # filter_set_ref=...,
            # fluor=...,
            # light_path=...,
            # light_source_settings=...,
            # nd_filter=...,
            # samples_per_pixel=...,
        )
        for c_idx, ch in enumerate(meta.channels or ())
    ]

    planes: list[m.Plane] = []
    for s_idx, loop_idx in zip(range(rdr._seq_count()), rdr.loop_indices()):
        fm = rdr.frame_metadata(s_idx)
        planes.extend(
            m.Plane(
                the_z=loop_idx.get(AXIS.Z, 0),
                the_t=loop_idx.get(AXIS.TIME, 0),
                the_c=c_idx,
                # exposure_time=...,
                # exposure_time_unit=...,
                delta_t=round(fm_ch.time.relativeTimeMs, 6),
                delta_t_unit=UnitsTime.MILLISECOND,  # default is "s"
                position_x=round(fm_ch.position.stagePositionUm.x, 6),
                position_y=round(fm_ch.position.stagePositionUm.y, 6),
                position_z=round(fm_ch.position.stagePositionUm.z, 6),
                position_x_unit=UnitsLength.MICROMETER,  # default is 'reference frame'
                position_y_unit=UnitsLength.MICROMETER,  # default is 'reference frame'
                position_z_unit=UnitsLength.MICROMETER,  # default is 'reference frame'
            )
            for c_idx, fm_ch in enumerate(fm.channels)
        )

    dims = "".join(reversed(list(f.sizes)))
    dim_order = next(
        (x for x in DimensionOrder if x.value.startswith(dims)), DimensionOrder.XYCZT
    )
    pixels = m.Pixels(
        id="Pixels:0",
        channels=channels,
        planes=planes,
        dimension_order=dim_order,
        type=str(f.dtype),
        significant_bits=f.attributes.bitsPerComponentSignificant,
        size_x=f.sizes.get(AXIS.X, 1),
        size_y=f.sizes.get(AXIS.Y, 1),
        size_z=f.sizes.get(AXIS.Z, 1),
        size_c=f.sizes.get(AXIS.CHANNEL, 1),
        size_t=f.sizes.get(AXIS.TIME, 1),
        physical_size_x=f.voxel_size().x,
        physical_size_y=f.voxel_size().y,
        physical_size_z=f.voxel_size().z,
        physical_size_x_unit=UnitsLength.MICROMETER,  # default is um
        physical_size_y_unit=UnitsLength.MICROMETER,  # default is um
        physical_size_z_unit=UnitsLength.MICROMETER,  # default is um
        metadata_only=True,
    )

    image = m.Image(
        id="Image:0",
        name=Path(f.path).stem,
        pixels=pixels,
        acquisition_date=rdr._acquisition_date(),
    )

    instrument = m.Instrument(
        id="Instrument:0",
        detectors=ome_detectors(rdr._cached_raw_metadata()),
        # TODO:
        # dichroics: List[Dichroic]
        # filter_sets: List[FilterSet]
        # filters: List[Filter]
        # light_source_group: List[LightSourceGroupType]
        # microscope: Optional[Microscope]
    )
    if ch0 is not None:
        scope = ch0.microscope
        instrument.objectives.append(
            m.Objective(
                id="Objective:0",
                nominal_magnification=scope.objectiveMagnification,
                lens_na=scope.objectiveNumericalAperture,
                # immersion=scope.ome_objective_immersion(),
            )
        )

    return m.OME(
        images=[image],
        creator=f"nd2 v{__version__}",
        instruments=[instrument],
    )


def ome_contrast_method(flags: list[ModalityFlags]) -> channel.ContrastMethod | None:
    """Return the ome_types ContrastMethod for this channel, if known."""
    # TODO: this is not exhaustive, and we need to check if a given
    # channel has flags for multiple contrast methods
    if "fluorescence" in flags:
        return ContrastMethod.FLUORESCENCE
    if "brightfield" in flags:
        return ContrastMethod.BRIGHTFIELD
    if "phaseContrast" in flags:
        return ContrastMethod.PHASE
    if "diContrast" in flags:
        return ContrastMethod.DIC
    return None


def ome_acquisition_mode(flags: list[ModalityFlags]) -> channel.AcquisitionMode | None:
    """Return the ome_types AcquisitionMode for this channel, if known."""
    # TODO: this is not exhaustive, and we need to check if a given
    # channel has flags for multiple contrast methods
    if "laserScanConfocal" in flags:
        return AcquisitionMode.LASER_SCANNING_CONFOCAL_MICROSCOPY
    if "spinningDiskConfocal" in flags:
        return AcquisitionMode.SPINNING_DISK_CONFOCAL
    if "sweptFieldConfocalPinhole" in flags:
        return AcquisitionMode.SWEPT_FIELD_CONFOCAL
    if "sweptFieldConfocalSlit" in flags:
        return AcquisitionMode.SLIT_SCAN_CONFOCAL
    if "brightfield" in flags:
        return AcquisitionMode.BRIGHT_FIELD
    if "SIM" in flags:
        return AcquisitionMode.STRUCTURED_ILLUMINATION
    if "TIRF" in flags:
        return AcquisitionMode.TOTAL_INTERNAL_REFLECTION
    if "multiphoton" in flags:
        return AcquisitionMode.MULTI_PHOTON_MICROSCOPY
    if "fluorescence" in flags:
        return AcquisitionMode.WIDE_FIELD
    return None


def ome_illumination_type(
    flags: list[ModalityFlags],
) -> channel.IlluminationType | None:
    """Return the ome_types IlluminationType for this channel, if known."""
    if "fluorescence" in flags:
        return IlluminationType.EPIFLUORESCENCE
    if "brightfield" in flags:
        return IlluminationType.TRANSMITTED
    if "multiphoton" in flags:
        return IlluminationType.NON_LINEAR
    return None


def ome_detectors(raw_meta: RawMetaDict) -> list[m.Detector]:
    pplanes = raw_meta.get("sPicturePlanes", {})
    sample_settings = pplanes.get("sSampleSetting", {})
    info: set[tuple[str | None, str | None]] = set()
    for ch_settings in sample_settings.values():
        camdict = ch_settings.get("pCameraSetting")
        if camdict:
            model = camdict.get("CameraFamilyName") or camdict.get("CameraUserName")
            info.add((model, camdict.get("CameraUniqueName")))

        # other info that could be added:
        # lot_number: Optional[str] = None
        # amplification_gain: Optional[float]
        # annotation_ref: List[AnnotationRef]
        # gain: Optional[float]
        # offset: Optional[float]
        # type_: Optional[Type]
        # voltage: Optional[float]
        # voltage_unit: Optional[UnitsElectricPotential]
        # zoom: Optional[float]

    return [
        m.Detector(id=f"Detector:{idx}", model=model, serial_number=sn)
        for idx, (model, sn) in enumerate(info)
    ]
