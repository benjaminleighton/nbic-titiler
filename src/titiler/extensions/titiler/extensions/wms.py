"""wms Extension."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List
from urllib.parse import urlencode
import base64
import json
from io import BytesIO


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

import jinja2
import numpy
import rasterio
import pyproj
from fastapi import Depends, HTTPException
from rasterio.crs import CRS
import rasterio.warp
from rio_tiler.mosaic import mosaic_reader
from rio_tiler.mosaic.methods.base import MosaicMethodBase
from starlette.requests import Request
from starlette.responses import Response
from starlette.templating import Jinja2Templates

from titiler.core.dependencies import ColorFormulaParams, RescalingParams
from titiler.core.factory import BaseTilerFactory, FactoryExtension
from titiler.core.resources.enums import ImageType, MediaType
from titiler.core.utils import render_image

jinja2_env = jinja2.Environment(
    loader=jinja2.ChoiceLoader([jinja2.PackageLoader(__package__, "templates")])
)
DEFAULT_TEMPLATES = Jinja2Templates(env=jinja2_env)

class WMSMediaType(str, Enum):
    """Responses Media types for WMS"""

    tif = "image/tiff; application=geotiff"
    jp2 = "image/jp2"
    png = "image/png"
    jpeg = "image/jpeg"
    jpg = "image/jpg"
    webp = "image/webp"


@dataclass
class OverlayMethod(MosaicMethodBase):
    """Overlay data on top."""

    def feed(self, array: numpy.ma.MaskedArray):
        """Add data to the mosaic array."""
        if self.mosaic is None:  # type: ignore
            self.mosaic = array

        else:
            pidex = self.mosaic.mask & ~array.mask

            mask = numpy.where(pidex, array.mask, self.mosaic.mask)
            self.mosaic = numpy.ma.where(pidex, array, self.mosaic)
            self.mosaic.mask = mask


@dataclass
class wmsExtension(FactoryExtension):
    """Add /wms endpoint to a TilerFactory."""

    supported_crs: List[str] = field(default_factory=lambda: ["EPSG:4326", "EPSG:3857"])
    supported_format: List[str] = field(
        default_factory=lambda: [
            "image/png",
            "image/jpeg",
            "image/jpg",
            "image/webp",
            "image/jp2",
            "image/tiff; application=geotiff",
        ]
    )
    supported_version: List[str] = field(
        default_factory=lambda: ["1.0.0", "1.1.1", "1.3.0"]
    )
    templates: Jinja2Templates = DEFAULT_TEMPLATES

    def register(self, factory: BaseTilerFactory):  # noqa: C901
        """Register endpoint to the tiler factory."""

        @factory.router.get(
            "/wms",
            response_class=Response,
            responses={
                200: {
                    "description": "Web Map Server responses",
                    "content": {
                        "application/xml": {},
                        "image/png": {},
                        "image/jpeg": {},
                        "image/jpg": {},
                        "image/webp": {},
                        "image/jp2": {},
                        "image/tiff; application=geotiff": {},
                    },
                },
            },
            openapi_extra={
                "parameters": [
                    {
                        "required": False,
                        "schema": {
                            "title": "Legend aspect ratio",
                            "type": "float",
                        },
                        "name": "legend_aspect_ratio",
                        "in": "query",
                    },
                    {
                        "required": True,
                        "schema": {
                            "title": "Request name",
                            "type": "string",
                            "enum": [
                                "GetCapabilities",
                                "GetMap",
                                "GetFeatureInfo",
                                "GetLegendGraphic"
                            ],
                        },
                        "name": "REQUEST",
                        "in": "query",
                    },
                    {
                        "required": False,
                        "schema": {
                            "title": "WMS Service type",
                            "type": "string",
                            "enum": ["wms"],
                        },
                        "name": "SERVICE",
                        "in": "query",
                    },
                    {
                        "required": False,
                        "schema": {
                            "title": "WMS Request version",
                            "type": "string",
                            "enum": [
                                "1.1.0",
                                "1.1.1",
                                "1.3.0",
                            ],
                        },
                        "name": "VERSION",
                        "in": "query",
                    },
                    {
                        "required": True,
                        "schema": {
                            "title": "Comma-separated list of one or more map layers."
                        },
                        "name": "LAYERS",
                        "in": "query",
                    },
                    {
                        "required": False,
                        "schema": {
                            "title": "Output format of service metadata/map",
                            "type": "string",
                            "enum": [
                                "text/html",
                                "application/xml",
                                "image/png",
                                "image/jpeg",
                                "image/jpg",
                                "image/webp",
                                "image/jp2",
                                "image/tiff; application=geotiff",
                            ],
                        },
                        "name": "FORMAT",
                        "in": "query",
                    },
                    {
                        "required": False,
                        "schema": {
                            "title": "Sequence number or string for cache control"
                        },
                        "name": "UPDATESEQUENCE",
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
                            "title": "Bounding box corners (lower left, upper right) in CRS units.",
                            "type": "string",
                        },
                        "name": "BBOX",
                        "in": "query",
                    },
                    {
                        "required": False,
                        "schema": {
                            "title": "Width in pixels of map picture.",
                            "type": "integer",
                        },
                        "name": "WIDTH",
                        "in": "query",
                    },
                    {
                        "required": False,
                        "schema": {
                            "title": "Height in pixels of map picture.",
                            "type": "integer",
                        },
                        "name": "HEIGHT",
                        "in": "query",
                    },
                    # Non-Used
                    {
                        "required": False,
                        "schema": {
                            "title": "Comma-separated list of one rendering style per requested layer."
                        },
                        "name": "STYLES",
                        "in": "query",
                    },
                    {
                        "required": False,
                        "schema": {
                            "title": "Background transparency of map (default=FALSE).",
                            "type": "boolean",
                            "default": False,
                        },
                        "name": "TRANSPARENT",
                        "in": "query",
                    },
                    # {
                    #     "required": False,
                    #     "schema": {
                    #         "title": "Hexadecimal red-green-blue colour value for the background color (default=FFFFFF).",
                    #         "type": "string",
                    #         "default": "FFFFFF",
                    #     },
                    #     "name": "BGCOLOR",
                    #     "in": "query",
                    # },
                    {
                        "required": False,
                        "schema": {
                            "title": "The format in which exceptions are to be reported by the WMS (default=JSON).",
                            "type": "string",
                            "enum": ["JSON"],
                        },
                        "name": "EXCEPTIONS",
                        "in": "query",
                    },
                    # {
                    #     "required": False,
                    #     "schema": {
                    #         "title": "Time value of layer desired.",
                    #         "type": "string",
                    #     },
                    #     "name": "TIME",
                    #     "in": "query",
                    # },
                    # {
                    #     "required": False,
                    #     "schema": {
                    #         "title": "Elevation of layer desired.",
                    #         "type": "string",
                    #     },
                    #     "name": "ELEVATION",
                    #     "in": "query",
                    # },
                ]
            },
        )
        def wms(  # noqa: C901
            request: Request,
            # vendor (titiler) parameters
            layer_params=Depends(factory.layer_dependency),
            dataset_params=Depends(factory.dataset_dependency),
            post_process=Depends(factory.process_dependency),
            rescale=Depends(RescalingParams),
            color_formula=Depends(ColorFormulaParams),
            colormap=Depends(factory.colormap_dependency),
            reader_params=Depends(factory.reader_dependency),
            env=Depends(factory.environment_dependency),
        ):
            """Return a WMS query for a single COG.

            GetCapability will generate a WMS XML definition.

            GetMap is mostly copied from titiler.core.factory.TilerFactory.part.part
            """
            req = {k.lower(): v for k, v in request.query_params.items()}

            # Request is mandatory
            request_type = req.get("request")
            if not request_type:
                raise HTTPException(
                    status_code=400, detail="Missing WMS 'REQUEST' parameter."
                )

            # layers must be base64 encoded
            if 'layers' in req:
                try:
                    layers = base64.b64decode(req['layers']).decode('utf-8')
                except:
                    layers = req['layers']#.decode('utf-8')
            elif 'layer' in req:
                try:
                    layers = base64.b64decode(req['layer']).decode('utf-8')
                except:
                    layers = req['layer']#.decode('utf-8')
            else:
                layers = ''
            
            if request_type != 'GetLegendGraphic':
                inlayers = layers #req.get("layers")
                layers=Depends(factory.path_dependency),
                if inlayers is None:
                    raise HTTPException(
                        status_code=400, detail="Missing WMS 'LAYERS' parameter."
                    )

                layers = list(inlayers.split(","))
                if not layers or not inlayers:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid 'LAYERS' parameter: {inlayers}.",
                    )
                
            def get_bounding_box(crs, bbox):
                # WMS 1.3.0 completely relies on how the CRS is defined
                # so, if the first axis defined in the CRS is a East-West one, we do nothing
                # this is because the bounding box is generally defined as [left, bottom, right top]
                # so, with the East-West axis being the first one, e.g. [LEFT, bottom, RIGHT, top]
                if pyproj.CRS(crs).axis_info[0].direction == 'east':
                    pass
                # otherwise we need to flip the order, to have [BOTTOM, left, TOP, right]
                else:
                    bbox = [
                        bbox[1],
                        bbox[0],
                        bbox[3],
                        bbox[2],
                    ]
                return bbox
            # GetMap: Return an image chip
            def get_map_data(req):
                # Required parameters:
                # - VERSION
                # - REQUEST=GetMap,
                # - LAYERS
                # - STYLES
                # - CRS
                # - BBOX
                # - WIDTH
                # - HEIGHT
                # - FORMAT
                # Optional parameters: TRANSPARENT, BGCOLOR, EXCEPTIONS, TIME, ELEVATION, ...

                # List of required parameters (styles and crs are excluded)
                req_keys = {
                    "version",
                    "request",
                    "layers",
                    "bbox",
                    "width",
                    "height",
                    "crs"
                }

                intrs = set(req.keys()).intersection(req_keys)
                missing_keys = req_keys.difference(intrs)
                if len(missing_keys) > 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing 'GetMap' parameters: {missing_keys}",
                    )

                version = req["version"]
                if version not in self.supported_version:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid 'VERSION' parameter: {version}. Allowed versions include: {self.supported_version}",
                    )

                if not set(req.keys()).intersection({"crs", "srs"}):
                    raise HTTPException(
                        status_code=400, detail="Missing 'CRS' or 'SRS parameters."
                    )

                crs_value = req.get("crs", req.get("srs"))
                if not crs_value:
                    raise HTTPException(
                        status_code=400, detail="Invalid 'CRS' parameter."
                    )

                crs = CRS.from_user_input(crs_value)

                bbox = list(map(float, req["bbox"].split(",")))
                if len(bbox) != 4:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid 'BBOX' parameters: {req['bbox']}. Needs 4 coordinates separated by commas",
                    )

                bbox = get_bounding_box(crs, bbox)

                if transparent := req.get("transparent", False):
                    if str(transparent).lower() == "true":
                        transparent = True

                    elif str(transparent).lower() == "false":
                        transparent = False

                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid 'TRANSPARENT' parameter: {transparent}. Should be one of ['FALSE', 'TRUE'].",
                        )

                format = 'None'
                if 'format' in req:
                    if req["format"] not in self.supported_format:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid 'FORMAT' parameter: {req['format']}. Should be one of {self.supported_format}.",
                        )
                    format = ImageType(WMSMediaType(req["format"]).name)

                height, width = int(req["height"]), int(req["width"])
                # raise RuntimeError


                def _reader(src_path: str):
                    with rasterio.Env(**env):
                        with factory.reader(src_path, **reader_params) as src_dst:
                            return src_dst.part(
                                bbox,
                                width=width,
                                height=height,
                                dst_crs=crs,
                                bounds_crs=crs,
                                **layer_params,
                                **dataset_params,
                            )

                image, assets_used = mosaic_reader(
                    layers,
                    _reader,
                    pixel_selection=OverlayMethod(),
                )


                return image, format, transparent

            def render_legend(colormap, labels, legend_type, rescale=(0,1), aspect_ratio=0.15):

                if not aspect_ratio:
                    # we have not specified a width or height, so we will generate a tight layout legend
                    # here we use default width/height
                    bbox_fig = 'tight'
                else:
                    # we have specified a width and height, so we will be strict about it
                    # this means that we need to manually calculate the Bbox
                    bbox_fig = None

                aspect_ratio = float(aspect_ratio)
                # get the initial settings
                # width /= 2.54
                # height /= 2.54
                dpi = 300

                # initialise the canvas
                fig = plt.figure(
                    figsize = (1.5/2.54, 10/2.54),
                    dpi=dpi,
                )
                ax = fig.add_subplot()
                # this is the memory buffer used to save the rendered image
                buf = BytesIO()

                def _enforce_aspect_ratio(fig, ax, aspect_ratio):
                    ox, oy, w, h = ax.get_tightbbox().transformed(fig.dpi_scale_trans.inverted()).bounds
                    current_aspect_ratio = w/h
                    wanted_aspect_ratio = aspect_ratio
                    factor = current_aspect_ratio/wanted_aspect_ratio

                    if factor > 1:
                        # we need to increase the height
                        new_h = h * factor
                        # we also need to offset the y origin to make sure that the empty space is added at the bottom
                        new_oy = oy - (new_h - h)
                        new_bbox = (ox, new_oy, w, new_h)
                    elif factor < 1:
                        # we need to increase the width
                        new_bbox = (ox, oy, w/factor, h)
                    else:
                        new_bbox = (ox, oy, w, h)
                    return mpl.transforms.Bbox.from_bounds(*new_bbox)


                if legend_type == 'linear':
                    # generate a linear colormap from the {value: rgba_code} dictionary
                    if not isinstance(rescale, list) or not isinstance(rescale[0], tuple) or len(rescale[0]) != 2:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Linear colormap legends require the use of a valid rescale parameter (e.g. [(1.0, 2.0)]). {rescale} was provided",
                        )

                    cmap = mpl.colors.LinearSegmentedColormap.from_list(
                        'custom',
                        [(k/255, [c/255 for c in rgba]) for (k, rgba) in colormap.items()],
                        256,
                    )
                    norm = mpl.colors.Normalize(vmin=rescale[0][0], vmax=rescale[0][1])
                    # this will create a colorbar in the axis, normalised between min/max
                    cb = mpl.colorbar.ColorbarBase(
                        ax, 
                        cmap=cmap,
                        norm=norm,
                        orientation='vertical'
                    )
                    if labels:
                        cb.set_label(labels)

                    if not bbox_fig:
                        bbox_fig = _enforce_aspect_ratio(fig, ax, aspect_ratio).padded(0.1)

                elif legend_type == 'interval':
                    # generate a interval colormap, like the one for contourf
                    bounds = []
                    colors = []
                    # this time we should receive a sequence [(min_interval, max_interval), rgba_code]
                    under=None
                    over=None
                    for b, c in colormap:
                        if b[0] == float('-inf'):
                            under = [c[0]/255, c[1]/255, c[2]/255] + [1]
                        elif b[1] == float("inf"):
                            over = [c[0]/255, c[1]/255, c[2]/255] + [1]
                        else:
                            bounds.append(b[0])
                            colors.append([c[0]/255, c[1]/255, c[2]/255] + [1])
                    # remember to close the last interval
                    if b[1] == float("inf"):
                        bounds.append(b[0])
                    else:
                        bounds.append(b[1])
                    # raise

                    if over and under:
                        extend='both'
                    elif over:
                        extend='max'
                    elif under:
                        extend='min'
                    else:
                        extend='neither'

                    cmap = mpl.colors.ListedColormap(colors).with_extremes(under=under, over=over)
                    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                    cb = mpl.colorbar.ColorbarBase(ax, 
                        cmap=cmap,
                        norm=norm,
                        boundaries=bounds,
                        ticks=bounds,
                        extend=extend,
                        # spacing='proportional',
                        orientation='vertical'
                    )
                    if labels:
                        cb.set_label(labels)
                    
                    if not bbox_fig:
                        bbox_fig = _enforce_aspect_ratio(fig, ax, aspect_ratio).padded(0.1)

                elif legend_type == 'discrete':

                    # generate a dictionary of {rgba_values: values_or_labels}
                    legend_table = {}

                    # as lists are mutable objects, they cannot be used as keys. 
                    # To fix that we convert them to strings, but remove the [] to easily reconvert them later to lists

                    for k, v in colormap.items():
                        try:
                            # try to retrieve the label
                            lbl = labels[k]
                        except KeyError:
                            # if not there fall back to value
                            lbl = str(k)
                        try:
                            # now store the label_or_value associated with the color code.
                            # if that color is used for multiple values, we already seen it
                            legend_table[str(v)[1:-1]] = legend_table[str(v)[1:-1]] + ', ' + lbl
                        except KeyError:
                            legend_table[str(v)[1:-1]] = lbl

                    # create the list of patches to render
                    legend_elements = []
                    for col, lbl in legend_table.items():

                        # the rgba color is reconverted to list using the str split function
                        legend_elements.append(Patch(facecolor=[int(c)/255 for c in col.split(', ')], edgecolor='k',
                                            label=lbl.capitalize()))

                    # render the legend
                    legend = ax.legend(handles=legend_elements, loc='center')
                    # remove the axis that is automatically added
                    plt.gca().set_axis_off()
                    # draw the canvas
                    fig.canvas.draw()
                    # now find the boundix box of the legend, so that we can remove the empty space
                    bbox_fig = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Could not understand the legend_type {legend_type}.",
                    )
                
                # save the colorbar/legend as a png figure.
                # this is done in-memory
                fig.savefig(buf, dpi="figure", format='png', bbox_inches=bbox_fig, transparent=True)
                # reset the buffer pointer to the first location (is it needed?)
                buf.seek(0)
                # re-read the buffer and close it
                out = buf.read()
                buf.close()
                # return the buffer values
                return out

            # GetCapabilities: Return a WMS XML
            if request_type.lower() == "getcapabilities":
                # Required parameters:
                # - SERVICE=WMS
                # - REQUEST=GetCapabilities
                # Optional parameters: VERSION, FORMAT, UPDATESEQUENCE

                # List of required parameters (layers is added for titiler)
                req_keys = {"service", "request", "layers"}

                intrs = set(req.keys()).intersection(req_keys)
                missing_keys = req_keys.difference(intrs)
                if len(missing_keys) > 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing 'GetCapabilities' parameters: {missing_keys}",
                    )

                if not req["service"].lower() == "wms":
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid 'SERVICE' parameter: {req['service']}. Only 'wms' is accepted",
                    )

                version = req.get("version", "1.3.0")
                if version not in self.supported_version:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid 'VERSION' parameter: {version}. Allowed versions include: {self.supported_version}",
                    )

                wms_url = factory.url_for(request, "wms")

                qs_key_to_remove = [
                    "service",
                    "request",
                    "layers",
                    "version",
                    "format",
                    "updatesequence",
                ]
                qs = [
                    (key, value)
                    for (key, value) in request.query_params._list
                    if key.lower() not in qs_key_to_remove
                ]
                if qs:
                    wms_url += f"?{urlencode(qs)}"

                # Grab information from each layer provided
                layers_dict: Dict[str, Any] = {}
                for layer in layers:
                    layer_encoded = base64.b64encode(layer.encode()).decode('utf-8')
                    layers_dict[layer_encoded] = {}
                    title = layer.split('?')[0].split('/')[-1]
                    layers_dict[layer_encoded]['title'] = title
                    with rasterio.Env(**env):
                        with factory.reader(layer, **reader_params) as src_dst:
                            if src_dst.crs is None:
                                layer_crs = f"EPSG:4326"
                            else:
                                try:
                                    layer_crs = f"EPSG:{src_dst.crs.to_epsg()}"
                                except:
                                    layer_crs = src_dst.crs.to_string()

                            layers_dict[layer_encoded]['CRSs'] = {}
                            layers_dict[layer_encoded]['CRSs'][layer_crs] = get_bounding_box(layer_crs, src_dst.bounds)
                            # print(src_dst)
                            # raise RuntimeError(src_dst)

                            for crs in self.supported_crs:
                                if crs == layer_crs:
                                    continue
                                layers_dict[layer_encoded]['CRSs'][crs] = get_bounding_box(
                                    crs, 
                                    rasterio.warp.transform_bounds(
                                        layer_crs,
                                        crs,
                                        *src_dst.bounds
                                    )
                                )

                            # self.supported_crs.append(f"EPSG:{src_dst.crs.to_epsg()}")

                            # layers_dict[layer_encoded]["bounds"] = src_dst.bounds
                            layers_dict[layer_encoded][
                                "bounds_wgs84"
                            ] = src_dst.geographic_bounds
                            layers_dict[layer_encoded][
                                "abstract"
                            ] = src_dst.info().model_dump_json()

                # Build information for the whole service
                minx, miny, maxx, maxy = zip(
                    *[layers_dict[layer]["bounds_wgs84"] for layer in layers_dict]
                )

                legend_url = wms_url + '&REQUEST=GetLegendGraphic'

                return self.templates.TemplateResponse(
                    f"wms_{version}.xml",
                    {
                        "request": request,
                        "request_url": wms_url,
                        "legend_url":legend_url,
                        "formats": self.supported_format,
                        "available_epsgs": layers_dict[layer_encoded]['CRSs'],
                        "layers_dict": layers_dict,
                        "service_dict": {
                            "xmin": min(minx),
                            "ymin": min(miny),
                            "xmax": max(maxx),
                            "ymax": max(maxy),
                        },
                    },
                    media_type=MediaType.xml.value,
                )
            elif request_type.lower() == "getmap":

                image, format, transparent = get_map_data(req)

                if post_process:
                    image = post_process(image)

                if rescale:
                    image.rescale(rescale)

                if color_formula:
                    image.apply_color_formula(color_formula)

                # if colormap:
                #     image = image.apply_colormap(colormap)

                content, media_type = render_image(
                    image,
                    output_format=format,
                    colormap=colormap,
                    add_mask=transparent,
                )
                return Response(content, media_type=media_type)
            elif request_type.lower() == "getfeatureinfo":
                # Required parameters:
                # - VERSION
                # - REQUEST=GetFeatureInfo
                # - LAYERS
                # - CRS or SRS
                # - WIDTH
                # - HEIGHT
                # - QUERY_LAYERS
                # - I (Pixel column)
                # - J (Pixel row)
                # Optional parameters: INFO_FORMAT, FEATURE_COUNT, ...

                req_keys = {
                    "version",
                    "request",
                    "layers",
                    "width",
                    "height",
                    "query_layers",
                    "i",
                    "j",
                }
               
                i = int(req['i'])
                j = int(req['j'])
                image, format, transparent = get_map_data(req)

                # Convert the sample value to XML
                html_content = f"{image.data[0, j, i]}\n"

                return Response(html_content, media_type="text/html")
            elif request_type.lower() == "getlegendgraphic":
                if not colormap:
                    raise HTTPException(
                        status_code=400,
                        detail=f"REQUEST {request_type} cannot be fulfilled as no colormap information was specified",
                    )
                
                # if a colormap name has been passed, that will be used to create a linear colorbar legend
                has_colormap_name = bool(req.get('colormap_name', False))
                if has_colormap_name:
                    legend_type = 'linear'
                else:
                    legend_type = req.get('colormap_type', 'linear')

                # parse the labels
                # if discrete, then labels should be a JSON encoded object
                if legend_type == 'discrete':
                    try:
                        labels = req.get('colormap_labels', '{}').replace("'", '"')
                        labels = json.loads(labels)
                    except json.JSONDecodeError:
                        raise HTTPException(
                            status_code=400, detail="Could not parse the colormap label value."
                    )
                    print(labels)
                    # the JSON encoded value should return either a {key:value} or [[key, value]]
                    if isinstance(labels, list):
                        labels = {float(k):v for (k, v) in labels}
                    elif isinstance(labels, dict):
                        labels = {float(k):v for k, v in labels.items()}
                    else:
                        raise HTTPException(
                            status_code=400, detail="colormap_label parameter is malformed. Expected {value:label} or [[value, label]]"
                        )
                else:
                    # otherwise we expect the labels to be the color ramp label
                    labels = req.get('colormap_labels', '')

                # render the legend
                legend = render_legend(colormap, labels, legend_type, rescale=rescale, aspect_ratio=req.get("legend_aspect_ratio", False))

                return Response(legend, media_type='image/png')

            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid 'REQUEST' parameter: {request_type}. Should be one of ['GetCapabilities', 'GetMap', 'GetFeatureInfo', 'GetLegendGraphic'].",
                )
