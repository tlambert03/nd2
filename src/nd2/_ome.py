from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from nd2 import __version__

from ._util import AXIS

try:
    import ome_types.model as m
    from ome_types.model.pixels import DimensionOrder
except ImportError:
    raise ImportError("ome-types is required to read OME metadata") from None

if TYPE_CHECKING:
    from nd2 import ND2File

    from ._pysdk._pysdk import ND2Reader as LatestSDKReader
    from .structures import (
        Metadata,
    )


def nd2_ome_metadata(nd2: ND2File) -> m.OME:
    if nd2.is_legacy:
        raise NotImplementedError("OME metadata is not available for legacy files")
    rdr = cast("LatestSDKReader", nd2._rdr)
    meta = cast("Metadata", nd2.metadata)

    ch0 = next(iter(meta.channels or ()), None)
    channels = [
        m.Channel(
            # id=...,
            name=ch.channel.name,
            acquisition_mode=ch.microscope.ome_acquisition_mode(),
            color=ch.channel.colorRGB,
            contrast_method=ch.microscope.ome_contrast_method(),
            pinhole_size=ch.microscope.pinholeDiameterUm,
            # pinhole_size_unit='um',  # default is "um"
            emission_wavelength=ch.channel.emissionLambdaNm,
            # emission_wavelength_unit='nm',  # default is "nm"
            excitation_wavelength=ch.channel.excitationLambdaNm,
            # excitation_wavelength_unit='nm',  # default is "nm"
            illumination_type=ch.microscope.ome_illumination_type(),
            # detector_settings=...,
            # filter_set_ref=...,
            # fluor=...,
            # light_path=...,
            # light_source_settings=...,
            # nd_filter=...,
            # samples_per_pixel=...,
        )
        for ch in (meta.channels or ())
    ]

    coord_axes = [x for x in nd2.sizes if x not in {AXIS.CHANNEL, AXIS.Y, AXIS.X}]
    # FIXME: this is so stupid... we go through a similar loop in many places
    # in the code.  fix it up to be more efficient.
    planes: list[m.Plane] = []
    for p_idx in range(rdr._seq_count()):
        coords = dict(zip(coord_axes, rdr._coords_from_seq_index(p_idx)))
        planes.extend(
            m.Plane(
                the_z=coords.get(AXIS.Z, 0),
                the_t=coords.get(AXIS.TIME, 0),
                the_c=c_idx,
                # exposure_time=...,
                # exposure_time_unit=...,
                # delta_t=...,
                # delta_t_unit=...,
                # position_x=...,
                # position_x_unit=...,
                # position_y=...,
                # position_y_unit=...,
                # position_z=...,
                # position_z_unit=...,
            )
            for c_idx in range(nd2.attributes.channelCount or 1)
        )

    dims = "".join(reversed(nd2.sizes))
    dim_order = next((x for x in DimensionOrder if x.value.startswith(dims)), None)
    pixels = m.Pixels(
        channels=channels,
        planes=planes,
        dimension_order=dim_order or DimensionOrder.XYCZT,
        type=str(nd2.dtype),
        significant_bits=nd2.attributes.bitsPerComponentSignificant,
        size_x=nd2.sizes.get(AXIS.X, 1),
        size_y=nd2.sizes.get(AXIS.Y, 1),
        size_z=nd2.sizes.get(AXIS.Z, 1),
        size_c=nd2.sizes.get(AXIS.CHANNEL, 1),
        size_t=nd2.sizes.get(AXIS.TIME, 1),
        physical_size_x=nd2.voxel_size().x,
        physical_size_y=nd2.voxel_size().y,
        physical_size_z=nd2.voxel_size().z,
        # physical_size_x_unit='um',  # default is um
        # physical_size_y_unit='um',  # default is um
        # physical_size_z_unit='um',  # default is um
    )

    image = m.Image(
        name=Path(nd2.path).stem,
        pixels=pixels,
        acquisition_date=rdr._acquisition_date(),
    )

    instruments: list[m.Instrument] = []
    if ch0 is not None:
        scope = ch0.microscope
        instruments.append(
            m.Instrument(
                objectives=[
                    m.Objective(
                        nominal_magnification=scope.objectiveMagnification,
                        lens_na=scope.objectiveNumericalAperture,
                        # immersion=scope.ome_objective_immersion(),
                    )
                ]
            )
        )

    return m.OME(images=[image], creator=f"nd2 v{__version__}", instruments=instruments)
