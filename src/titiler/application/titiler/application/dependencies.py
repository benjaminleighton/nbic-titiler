"""dependencies.

app/dependencies.py

"""

import json
from enum import Enum
from typing import Dict, Optional, Union, Sequence, List, Annotated

import numpy
import matplotlib
from rio_tiler.colormap import cmap, parse_color
from fastapi import HTTPException, Query


ColorMapName = Enum(  # type: ignore
    "ColorMapName", [(a, a) for a in sorted(cmap.list())]
)

class ColorMapType(str, Enum):
    """Colormap types."""

    discrete = "discrete"
    linear = "linear"
    interval = "interval"


def ColorMapParams(
    colormap_name: ColorMapName = Query(None, description="Colormap name"),
    colormap: str = Query(None, description="JSON encoded {value:color} or [[value, color]] mapping. For linear type values will be interpolated, for interval type values will be consider the start of the interval"),
    colormap_type: ColorMapType = Query(ColorMapType.linear, description="User input colormap type (used only with custom Colormap)"),
    colormap_labels: Annotated[Optional[str], Query(description="JSON encoded {value:label} or [[value, label]] mapping (for discrete Colormap), or a single label as string (for linear and interval Colormap)")] = None
) -> Optional[Union[Dict, Sequence]]:
    """Colormap Dependency."""
    if colormap_name:
        return cmap.get(colormap_name.value)

    if colormap:
        try:
            cm = json.loads(
                colormap,
                # object_hook=lambda x: {float(k): parse_color(v) for k, v in x.items()},
            )
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, detail="Could not parse the colormap value."
            )
        
        if isinstance(cm, list):
            try:
                cm = {float(k):parse_color(v) for (k, v) in cm}
            except TypeError:
                # passed colormap using the old way
                # this is [[min, max], [RGB values]]
                # vs [max, [RGB values]]
                cm = {float(max(k)):parse_color(v) for (k, v) in cm}
        elif isinstance(cm, dict):
            cm = {float(k):parse_color(v) for k, v in cm.items()}
        else:
            raise HTTPException(
                status_code=400, detail="Colormap parameter is malformed. Expected {value:color} or [[value, color]]"
            )

        if colormap_type == ColorMapType.linear:
            # if linear, we need to enforce rescaling between min and max
            min_val = min(list(cm.keys()))
            max_val = max(list(cm.keys()))
            n_vals = 256
            cm = matplotlib.colors.LinearSegmentedColormap.from_list(
                'custom',
                [
                    ((k - min_val)/(max_val - min_val), matplotlib.colors.to_hex([v / 255 for v in rgba]))
                    for (k, rgba) in cm.items()
                ],
                n_vals,
            )
            x = numpy.linspace(0, 1, n_vals)
            cmap_vals = cm(x)[:, :]
            cmap_uint8 = (cmap_vals * 255).astype('uint8')
            cm_out = {int(idx): value.tolist() for idx, value in enumerate(cmap_uint8)}
        elif colormap_type == ColorMapType.interval:
            # if interval, we need to provide a list of tuples ([idx_start, idx_end], [color sequence])
            intervals = sorted(list(cm.keys()))
            cm_out = []
            for k_start, k_end in zip(intervals[:-1], intervals[1:]):
                cm_out.append(([k_start, k_end], cm[k_start]))
        elif colormap_type == ColorMapType.discrete:
            # for the discrete case, that is a pass-through
            cm_out = cm

        else:
            raise HTTPException(
                status_code=400, detail="Could not understand the colormap type"
            )

        return cm_out

    return None