"""wcs Extension."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode
import base64

import jinja2
import numpy
import rasterio
import pyproj
from attrs import define
from fastapi import Depends, HTTPException
from rasterio.crs import CRS
from rio_tiler.constants import WGS84_CRS
from rio_tiler.models import ImageData
from rio_tiler.mosaic import mosaic_reader
from rio_tiler.mosaic.methods.base import MosaicMethodBase
from starlette.requests import Request
from starlette.responses import Response
from starlette.templating import Jinja2Templates
from titiler.core.utils import render_image

from titiler.core.dependencies import (
    ColorFormulaParams,
    RescalingParams,
)
from titiler.core.factory import FactoryExtension, BaseTilerFactory
from titiler.core.resources.enums import ImageType, MediaType

jinja2_env = jinja2.Environment(
    loader=jinja2.ChoiceLoader([jinja2.PackageLoader(__package__, "templates")])
)
DEFAULT_TEMPLATES = Jinja2Templates(env=jinja2_env)


class CoverageMediaType(str, Enum):
    """Media types for WCS coverages (examples below)."""

    tif = "GeoTIFF"  # Simplified for demonstration
    # You could add more, e.g. 'application/x-netcdf', 'image/jp2', etc.


@dataclass
class OverlayMethod(MosaicMethodBase):
    """Overlay mosaic method, identical to WMS example."""

    def feed(self, array: numpy.ma.MaskedArray):
        if self.mosaic is None:  # type: ignore
            self.mosaic = array
        else:
            pidex = self.mosaic.mask & ~array.mask
            mask = numpy.where(pidex, array.mask, self.mosaic.mask)
            self.mosaic = numpy.ma.where(pidex, array, self.mosaic)
            self.mosaic.mask = mask


@dataclass
class wcsExtension(FactoryExtension):
    """
    A minimal WCS extension that creates a `/wcs` endpoint supporting:
    - GetCapabilities
    - DescribeCoverage
    - GetCoverage
    """

    supported_crs: List[str] = field(default_factory=lambda: ["EPSG:4326", "EPSG:3857"])
    supported_format: List[str] = field(
        default_factory=lambda: [
            "GeoTIFF",
        ]
    )
    # Example WCS versions. Adjust as needed or expand to handle more versions.
    supported_version: List[str] = field(default_factory=lambda: ["1.0.0","2.0.1"])

    templates: Jinja2Templates = DEFAULT_TEMPLATES

    def register(self, factory: BaseTilerFactory):
        """Register endpoint to the tiler factory."""

        @factory.router.get(
            "/wcs",
            response_class=Response,
            responses={
                200: {
                    "description": "Web Coverage Service responses",
                    "content": {
                        "application/xml": {},
                        "GeoTIFF": {},
                    },
                },
            },
            openapi_extra={
                "parameters": [
                    {
                        "required": True,
                        "schema": {
                            "title": "Request name",
                            "type": "string",
                            "enum": [
                                "GetCapabilities",
                                "DescribeCoverage",
                                "GetCoverage",
                            ],
                        },
                        "name": "REQUEST",
                        "in": "query",
                    },
                    {
                        "required": False,
                        "schema": {
                            "title": "WCS Service type",
                            "type": "string",
                            "default": "WCS",
                            "enum": ["WCS"],
                        },
                        "name": "SERVICE",
                        "in": "query",
                    },
                    {
                        "required": False,
                        "schema": {
                            "title": "WCS Request version",
                            "type": "string",
                            "default": "1.0.0",
                            "enum": [
                                "1.0.0",
                                "2.0.1",
                            ],
                        },
                        "name": "VERSION",
                        "in": "query",
                    },
                    {
                        "required": False,
                        "schema": {
                            "title": "Comma-separated coverage IDs",
                            "type": "string",
                        },
                        "name": "COVERAGE",
                        "in": "query",
                    },
                    {
                        "required": False,
                        "schema": {
                            "title": "Other way to specify coverages",
                            "type": "string",
                        },
                        "name": "LAYER",
                        "in": "query",
                    },
                    {
                        "required": False,
                        "schema": {
                            "title": "Bounding box corners (lower left, upper right) in CRS units.",
                            "type": "string",
                        },
                        "name": "BBOX",
                        "in": "query",
                    },
                    {
                        "required": False,
                        "schema": {
                            "title": "Coordinate reference system.",
                            "type": "string",
                        },
                        "name": "CRS",
                        "in": "query",
                    },
                    {
                        "required": False,
                        "schema": {
                            "title": "Width in pixels of coverage output (for demonstration).",
                            "type": "integer",
                        },
                        "name": "WIDTH",
                        "in": "query",
                    },
                    {
                        "required": False,
                        "schema": {
                            "title": "Height in pixels of coverage output (for demonstration).",
                            "type": "integer",
                        },
                        "name": "HEIGHT",
                        "in": "query",
                    },
                    {
                        "required": False,
                        "schema": {
                            "title": "Requested output coverage format",
                            "type": "string",
                            "enum": [
                                "GeoTIFF",
                            ],
                        },
                        "name": "FORMAT",
                        "in": "query",
                    },
                    # You can add more WCS parameters here (SUBSET=, RESX=, RESY=, TIME=, ELEVATION=, etc.)
                ]
            },
        )
        def wcs(
            request: Request,
            # Below are dependencies from titiler for reading the dataset
            reader_params=Depends(factory.reader_dependency),
            layer_params=Depends(factory.layer_dependency),
            dataset_params=Depends(factory.dataset_dependency),
            # post_process=Depends(factory.process_dependency),
            # rescale=Depends(RescalingParams),
            # color_formula=Depends(ColorFormulaParams),
            # colormap=Depends(factory.colormap_dependency),
            env=Depends(factory.environment_dependency),
        ):
            """
            WCS operations:
            - GetCapabilities: Return WCS capabilities document.
            - DescribeCoverage: Return coverage descriptions.
            - GetCoverage: Return coverage data (e.g., GeoTIFF).
            """

            # Collect query parameters (case-insensitive)
            req = {k.lower(): v for k, v in request.query_params.items()}

            # WCS requires SERVICE=WCS (in theory), but many clients omit or do different checks
            request_type = req.get("request", None)
            if not request_type:
                raise HTTPException(
                    status_code=400, detail="Missing WCS parameter 'REQUEST'."
                )

            layers = list(set(req.get("coverage", "").split(",") + req.get("layer", "").split(",")))

            # decode b64 encoded paths
            coverage_ids = []
            for c_id in layers:
                try:
                    coverage_ids.append(base64.b64decode(c_id).decode('utf-8'))
                except:
                    coverage_ids.append(c_id)

            # For demonstration, treat coverage IDs like "layers" in WMS
            # We can simply pass them to mosaic_reader or single COG readers.
            # In a real WCS, you'd have a separate coverage registry or config.

            # WCS "GetCapabilities"
            if request_type.lower() == "getcapabilities":
                version = req.get("version", "1.0.0")
                if version not in self.supported_version:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Invalid VERSION={version}, must be one of "
                            f"{self.supported_version}"
                        ),
                    )

                wcs_url = factory.url_for(request, "wcs")

                # Remove standard WCS keys from the query string so we can pass the rest along
                skip_keys = {"service", "request", "coverage", "layer", "version", "format"}
                qs = [
                    (key, value)
                    for (key, value) in request.query_params._list
                    if key.lower() not in skip_keys
                ]
                if qs:
                    wcs_url += f"?{urlencode(qs)}"

                # Gather coverage bounding box from each coverage
                coverages_info: Dict[str, Any] = {}
                for cov_path in coverage_ids:
                    if not cov_path:
                        continue
                    cov_id = cov_path.split('?')[0].split('/')[-1]
                    with rasterio.Env(**env):
                        print(reader_params)
                        with factory.reader(cov_path, **reader_params) as src:
                            # We'll store minimal bounding box and CRS info
                            if src.crs is None:
                                layer_crs = f"EPSG:4326"
                            else:
                                try:
                                    layer_crs = f"EPSG:{src.crs.to_epsg()}"
                                except:
                                    layer_crs = src.crs.to_string()

                            if layer_crs not in self.supported_crs:
                                self.supported_crs.append(layer_crs)

                            coverages_info[cov_id] = {
                                "crs": layer_crs,
                                "layer":base64.b64encode(cov_path.encode()).decode('utf-8'),
                                # "bounds": src.bounds,
                                "bounds_wgs84": src.geographic_bounds,
                                "metadata": src.info().model_dump_json(),
                            }

                # Build overall bounding box in WGS84 for the entire service
                if coverages_info:
                    minx, miny, maxx, maxy = zip(
                        *[
                            cinfo["bounds_wgs84"]
                            for cinfo in coverages_info.values()
                            if "bounds_wgs84" in cinfo
                        ]
                    )
                    service_bbox = {
                        "xmin": min(minx),
                        "ymin": min(miny),
                        "xmax": max(maxx),
                        "ymax": max(maxy),
                    }
                else:
                    service_bbox = {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}

                return self.templates.TemplateResponse(
                    name=f"wcs_{version}.xml",  # Provide your own Jinja2 template
                    context={
                        "request": request,
                        "request_url": wcs_url,
                        "formats": self.supported_format,
                        "coverages": coverages_info,
                        "service_bbox": service_bbox,
                        "available_epsgs": self.supported_crs,
                    },
                    media_type=MediaType.xml.value,
                )

            # WCS "DescribeCoverage"
            elif request_type.lower() == "describecoverage":
                version = req.get("version", "2.0.1")
                if version not in self.supported_version:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Invalid VERSION={version}, must be one of "
                            f"{self.supported_version}"
                        ),
                    )

                # For each coverage requested, gather metadata
                coverages_info: Dict[str, Any] = {}
                for cov_path in coverage_ids:
                    if not cov_path:
                        continue
                    cov_id = cov_path.split('?')[0].split('/')[-1]
                    with rasterio.Env(**env):
                        with factory.reader(cov_path, **reader_params) as src:
                            if src.crs is None:
                                layer_crs = f"EPSG:4326"
                            else:
                                try:
                                    layer_crs = f"EPSG:{src.crs.to_epsg()}"
                                except:
                                    layer_crs = src.crs.to_string()

                            if layer_crs not in self.supported_crs:
                                self.supported_crs.append(layer_crs)

                            axis_info = pyproj.CRS(src.crs).axis_info

                            axis_labels = []
                            axis_units = []
                            axis_shape = []
                            lower_corner = []
                            upper_corner = []

                            for axs in axis_info:
                                if not axs.abbrev:
                                    if axs.unit_name == 'metre':
                                        if axs.direction == 'east':
                                            axis_labels.append('x')
                                        else:
                                            axis_labels.append('y')
                                    elif axs.unit_name == 'degree':
                                        if axs.direction == 'east':
                                            axis_labels.append('Lon')
                                        else:
                                            axis_labels.append('Lat')
                                else:
                                    axis_labels.append(axs.abbrev)
                                axis_units.append(axs.unit_name)

                                if axs.direction == 'east':
                                    axis_shape.append(src.dataset.width -1)
                                    lower_corner.append(src.bounds[0])
                                    upper_corner.append(src.bounds[2])

                                elif axs.direction == 'north':
                                    axis_shape.append(src.dataset.height -1)
                                    lower_corner.append(src.bounds[1])
                                    upper_corner.append(src.bounds[3])
                                else:
                                    raise ValueError(f"Could not understand axis direction {axs.direction}")


                            coverages_info[cov_id] = {
                                "crs": layer_crs,
                                "axis_shape":axis_shape,
                                "axis_labels":axis_labels,
                                "axis_units":axis_units,
                                "nodata":src.dataset.nodata or '',
                                "layer":base64.b64encode(cov_path.encode()).decode('utf-8'),
                                "bounds_wgs84": src.geographic_bounds,
                                "resolution": src.dataset.res,
                                # "bounds": src.bounds,
                                "upper_corner": upper_corner,
                                "lower_corner": lower_corner,
                                "metadata": src.info().model_dump_json(),
                            }

                return self.templates.TemplateResponse(
                    name=f"wcs_{version}_describecoverage.xml",  # Provide your own Jinja2 template
                    context={
                        "request": request,
                        "coverages": coverages_info,
                        "available_epsgs": self.supported_crs,
                    },
                    media_type=MediaType.xml.value,
                )

            # WCS "GetCoverage"
            elif request_type.lower() == "getcoverage":
                version = req.get("version", "2.0.1")
                if version not in self.supported_version:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Invalid VERSION={version}, must be one of "
                            f"{self.supported_version}"
                        ),
                    )

                crs_value = req.get("crs", None)
                if not crs_value:
                    raise HTTPException(
                        status_code=400, detail="Missing 'CRS' parameter in GetCoverage."
                    )
                crs = CRS.from_user_input(crs_value)

                if "bbox" not in req:
                    raise HTTPException(
                        status_code=400,
                        detail="Missing 'BBOX' parameter in GetCoverage.",
                    )
                bbox_str = req["bbox"]
                print(bbox_str)

                try:
                    bbox = list(map(float, bbox_str.split(",")))
                    if len(bbox) != 4:
                        raise ValueError("BBOX must contain exactly four numeric values.")
                    # if all([b == 0 for b in bbox]):
                    #     raise ValueError("BBOX provided containst only zeros")
                except Exception:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid BBOX: {bbox_str}",
                    )

                print(bbox)

                width = int(req.get("width", 256))
                height = int(req.get("height", 256))

                fmt = req.get("format", self.supported_format[0])
                if fmt not in self.supported_format:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"Invalid coverage FORMAT: {fmt}, must be one of "
                            f"{self.supported_format}"
                        ),
                    )
                fmt = ImageType(CoverageMediaType(fmt).name)

                # Minimal mosaic reading from multiple coverages:
                def _reader(src_id: str):
                    with rasterio.Env(**env):
                        with factory.reader(src_id, **reader_params) as src_dst:
                            return src_dst.part(
                                bbox,
                                width=width,
                                height=height,
                                dst_crs=crs,
                                bounds_crs=crs,
                                **layer_params,
                                **dataset_params,
                            )

                coverage_ids_nonempty = [cov for cov in coverage_ids if cov]
                if not coverage_ids_nonempty:
                    raise HTTPException(
                        status_code=400,
                        detail="No valid coverage ID found in COVERAGE parameter.",
                    )

                image, assets_used = mosaic_reader(
                    coverage_ids_nonempty,
                    _reader,
                    pixel_selection=OverlayMethod(),
                )
                # raise

                print(bbox)
                print(image.data.shape)
                print(width, height)

                content, media_type = render_image(
                    image,
                    output_format=fmt,
                    add_mask=False
                )
                return Response(content, media_type=media_type)

            else:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid WCS 'REQUEST' parameter: {request_type}. "
                        "Expecting one of ['GetCapabilities', 'DescribeCoverage', 'GetCoverage']."
                    ),
                )
