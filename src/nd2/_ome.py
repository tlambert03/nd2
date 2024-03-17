from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import ome_types.model as m
from ome_types.model import Channel_AcquisitionMode as AcquisitionMode
from ome_types.model import Channel_ContrastMethod as ContrastMethod
from ome_types.model import Channel_IlluminationType as IlluminationType
from ome_types.model import Pixels_DimensionOrder as DimensionOrder
from ome_types.model import UnitsLength, UnitsTime

from nd2 import __version__

from ._util import AXIS

if TYPE_CHECKING:
    from nd2 import ND2File

    from ._sdk_types import RawMetaDict
    from .readers import ModernReader
    from .structures import ModalityFlags

# TODO: add support for splitting positions into separate files, and pyramids
# https://docs.openmicroscopy.org/ome-model/6.3.1/ome-tiff/specification.html#embedded-ome-xml-metadata
# https://docs.openmicroscopy.org/ome-model/6.3.1/ome-tiff/specification.html#supported-resolutions


def nd2_ome_metadata(
    f: ND2File, *, include_unstructured: bool = True, tiff_file_name: str | None = None
) -> m.OME:
    """Generate an [`ome_types.OME`][] object for an ND2 file.

    Parameters
    ----------
    f : ND2File
        The ND2 file.
    include_unstructured : bool
        Whether to include all available metadata in the OME file. If `True`, (the
        default), the `unstructured_metadata` method is used to fetch all retrievable
        metadata, and the output is added to OME.structured_annotations, where each key
        is the chunk key, and the value is a JSON-serialized dict of the metadata. If
        `False`, only metadata which can be directly added to the OME data model are
        included.
    tiff_file_name : str | None
        If provided, [`ome_types.model.TiffData`][] block entries are added for each
        [`ome_types.model.Plane`][] in the OME object, with the
        `TiffData.uuid.file_name` set to this value.  (Useful for exporting to
        tiff.)
    """
    if f.is_legacy:
        raise NotImplementedError("OME metadata is not available for legacy files")

    rdr = cast("ModernReader", f._rdr)
    meta = f.metadata
    images = []
    acquisition_date = rdr._acquisition_date()
    uuid_ = f"urn:uuid:{uuid.uuid4()}"
    sizes = dict(f.sizes)
    n_positions = sizes.pop(AXIS.POSITION, 1)
    loop_indices = rdr.loop_indices()
    voxel_size = f.voxel_size()
    _dims, shape = zip(*sizes.items())
    dims = "".join(reversed(_dims)).upper()
    dim_order = next(
        (x for x in DimensionOrder if x.value.startswith(dims)), DimensionOrder.XYCZT
    )

    # if sizes.get(AXIS.CHANNEL, 1) > 1 and sizes.get(AXIS.RGB, 1) > 1:
    #     warn("multi-channel RGB images are not well supported in nd2 OME metadata.")

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

    ch0 = next(iter(meta.channels or ()), None)
    channels = []
    for c_idx, ch in enumerate(meta.channels or ()):
        channel = m.Channel(
            id=f"Channel:{c_idx}",
            name=ch.channel.name,
            acquisition_mode=ome_acquisition_mode(ch.microscope.modalityFlags),
            contrast_method=ome_contrast_method(ch.microscope.modalityFlags),
            pinhole_size=ch.microscope.pinholeDiameterUm,
            pinhole_size_unit=UnitsLength.MICROMETER,  # default is "um"
            illumination_type=ome_illumination_type(ch.microscope.modalityFlags),
            samples_per_pixel=sizes.get(AXIS.RGB, 1),
            # detector_settings=...,
            # filter_set_ref=...,
            # fluor=...,
            # light_path=...,
            # light_source_settings=...,
            # nd_filter=...,
        )
        if not f.is_rgb:
            # if you include any of this for RGB images, the Bioformats OMETiffReader
            # will show all three RGB channels with the same color
            channel.color = m.Color(ch.channel.color)
            channel.emission_wavelength = ch.channel.emissionLambdaNm
            channel.emission_wavelength_unit = UnitsLength.NANOMETER
            channel.excitation_wavelength = ch.channel.excitationLambdaNm
            channel.excitation_wavelength_unit = UnitsLength.NANOMETER
        channels.append(channel)

    for p in range(n_positions):
        planes: list[m.Plane] = []
        tiff_blocks: list[m.TiffData] = []
        ifd = 0
        for s_idx, loop_idx in enumerate(loop_indices):
            if loop_idx.get(AXIS.POSITION, 0) != p:
                continue
            f_meta = rdr.frame_metadata(s_idx)
            for c_idx, fm_ch in enumerate(f_meta.channels):
                # TODO: i think RGB might actually need to be 3 planes with 1 spp
                z_idx = loop_idx.get(AXIS.Z, 0)
                t_idx = loop_idx.get(AXIS.TIME, 0)
                planes.append(
                    m.Plane(
                        the_z=z_idx,
                        the_t=t_idx,
                        the_c=c_idx,
                        # exposure_time=...,
                        # exposure_time_unit=...,
                        delta_t=round(fm_ch.time.relativeTimeMs, 6),
                        delta_t_unit=UnitsTime.MILLISECOND,  # default is "s"
                        position_x=round(fm_ch.position.stagePositionUm.x, 6),
                        position_y=round(fm_ch.position.stagePositionUm.y, 6),
                        position_z=round(fm_ch.position.stagePositionUm.z, 6),
                        position_x_unit=UnitsLength.MICROMETER,
                        position_y_unit=UnitsLength.MICROMETER,
                        position_z_unit=UnitsLength.MICROMETER,
                    )
                )
                if tiff_file_name is not None:
                    tiff_blocks.append(
                        m.TiffData(
                            uuid=m.TiffData.UUID(value=uuid_, file_name=tiff_file_name),
                            ifd=ifd,
                            first_c=c_idx,
                            first_t=t_idx,
                            first_z=z_idx,
                            plane_count=1,
                        )
                    )
                ifd += 1

        md_only = None if tiff_blocks else m.MetadataOnly()
        pixels = m.Pixels(
            id=f"Pixels:{p}",
            channels=channels,
            planes=planes,
            tiff_data_blocks=tiff_blocks,
            metadata_only=md_only,
            dimension_order=dim_order,
            type=str(f.dtype),
            significant_bits=f.attributes.bitsPerComponentSignificant,
            size_x=f.sizes.get(AXIS.X, 1),
            size_y=f.sizes.get(AXIS.Y, 1),
            size_z=f.sizes.get(AXIS.Z, 1),
            size_c=f.sizes.get(AXIS.CHANNEL, 1),
            size_t=f.sizes.get(AXIS.TIME, 1),
            physical_size_x=voxel_size.x,
            physical_size_y=voxel_size.y,
            physical_size_x_unit=UnitsLength.MICROMETER,
            physical_size_y_unit=UnitsLength.MICROMETER,
        )
        if AXIS.Z in sizes:
            pixels.physical_size_z = voxel_size.z
            pixels.physical_size_z_unit = UnitsLength.MICROMETER

        images.append(
            m.Image(
                instrument_ref=m.InstrumentRef(id=instrument.id),
                # objective_settings=...
                id=f"Image:{p}",
                name=Path(f.path).stem + (f" (Series {p})" if n_positions > 1 else ""),
                pixels=pixels,
                acquisition_date=acquisition_date,
            )
        )

    if ch0 is not None:
        scope = ch0.microscope
        instrument.objectives.append(
            m.Objective(
                id="Objective:0",
                nominal_magnification=scope.objectiveMagnification,
                lens_na=scope.objectiveNumericalAperture,
                # model=... # something like "Plan Fluor 10x Ph1 DLL"
                # immersion=scope.ome_objective_immersion(),
            )
        )

    ome = m.OME(images=images, creator=f"nd2 v{__version__}", instruments=[instrument])

    if include_unstructured:
        all_meta = m.MapAnnotation(
            description="ND2 unstructured metadata, encoded as a JSON string. "
            "Each key in this MapAnnotation is the name of a metadata chunk found in "
            "the ND2 file, and the value is the JSON-encoded data for that chunk.",
            namespace="https://github.com/tlambert03/nd2",
            value={
                k: json.dumps(v, default=_default_encoder)
                for k, v in rdr.unstructured_metadata().items()
            },
        )
        ome.structured_annotations = m.StructuredAnnotations(map_annotations=[all_meta])

    return ome


def _default_encoder(obj: Any) -> Any:
    if isinstance(obj, bytearray):
        return obj.decode("utf-8")
    return str(obj)  # pragma: no cover


def ome_contrast_method(flags: list[ModalityFlags]) -> ContrastMethod | None:
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


def ome_acquisition_mode(flags: list[ModalityFlags]) -> AcquisitionMode | None:
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


def ome_illumination_type(flags: list[ModalityFlags]) -> IlluminationType | None:
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
