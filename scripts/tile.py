import os
import math
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_origin
from rasterio.crs import CRS
from scipy.ndimage import uniform_filter

def refined_lee(img, size=7):
    img = img.astype('float32')
    mean = uniform_filter(img, size=size)
    mean_sq = uniform_filter(img * img, size=size)
    var = mean_sq - mean * mean
    sigma_sq = np.mean(var)
    wl = var / (var + sigma_sq)
    return mean + wl * (img - mean)


def _get_utm_crs_from_bounds(bounds, src_crs):
    if not src_crs.is_geographic:
        return src_crs
    lon = (bounds.left + bounds.right) / 2.0
    lat = (bounds.top + bounds.bottom) / 2.0
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return CRS.from_epsg(epsg)


def tile_sar_and_optical(
    sar_tif_path: str,
    optical_tif_path: str,
    tile_width: int,
    tile_height: int,
    pixel_size_meters: float
) -> None:
    print(f"Starting tiling for SAR='{sar_tif_path}', optical='{optical_tif_path}'")

    with rasterio.open(sar_tif_path) as sar_src:
        orig_crs = sar_src.crs
        orig_transform = sar_src.transform
        orig_bounds = sar_src.bounds
        sar_count = sar_src.count
        sar_dtype = sar_src.dtypes[0]
        sar_nodata = sar_src.nodata

        target_crs = _get_utm_crs_from_bounds(orig_bounds, orig_crs)
        if target_crs != orig_crs:
            print(f"  Reprojecting SAR from {orig_crs.to_string()} to {target_crs.to_string()} for meter-based tiling.")
        else:
            print(f"  SAR already in projected CRS: {orig_crs.to_string()}")

        dst_transform, width, height = calculate_default_transform(
            orig_crs, target_crs,
            sar_src.width, sar_src.height,
            *sar_src.bounds,
            resolution=pixel_size_meters
        )

        print(f"  Target SAR resolution: {pixel_size_meters} meters per pixel → tile grid will use that.")
        print(f"  Reprojected SAR dimensions (pixels): width={width}, height={height}")

        sar_reproj = np.zeros((sar_count, height, width), dtype='float32')
        reproject(
            source=rasterio.band(sar_src, list(range(1, sar_count + 1))),
            destination=sar_reproj,
            src_transform=orig_transform,
            src_crs=orig_crs,
            dst_transform=dst_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
        )

        print("  Applying Refined Lee filter to reprojected SAR...")
        for b in range(sar_reproj.shape[0]):
            print(f"    Filtering band {b+1}/{sar_reproj.shape[0]}")
            sar_reproj[b] = refined_lee(sar_reproj[b], size=7)

        left = dst_transform.c
        top = dst_transform.f
        right = left + dst_transform.a * width
        bottom = top + dst_transform.e * height
        sar_proj_bounds = rasterio.coords.BoundingBox(
            left=left,
            bottom=bottom,
            right=right,
            top=top
        )
        sar_grid_crs = target_crs
        sar_grid_transform = dst_transform

    tile_size_x = tile_width * pixel_size_meters
    tile_size_y = tile_height * pixel_size_meters

    grid_w = sar_proj_bounds.right - sar_proj_bounds.left
    grid_h = sar_proj_bounds.top - sar_proj_bounds.bottom

    tiles_across = math.ceil(grid_w / tile_size_x)
    tiles_down = math.ceil(grid_h / tile_size_y)

    print(f"Computed grid: {tiles_across} tiles across × {tiles_down} tiles down (in meters).")

    sar_out_dir = os.path.splitext(os.path.basename(sar_tif_path))[0] + "_tiles"
    opt_out_dir = os.path.splitext(os.path.basename(optical_tif_path))[0] + "_tiles"
    os.makedirs(sar_out_dir, exist_ok=True)
    os.makedirs(opt_out_dir, exist_ok=True)

    with rasterio.open(optical_tif_path) as opt_src:
        opt_crs = opt_src.crs
        opt_transform = opt_src.transform
        opt_count = opt_src.count
        opt_dtype = opt_src.dtypes[0]
        opt_nodata = opt_src.nodata

        print(f"  Optical CRS: {opt_crs.to_string()}, will reproject per tile into {sar_grid_crs.to_string()}.")

        for col in range(tiles_across):
            for row in range(tiles_down):
                tlx = sar_proj_bounds.left + col * tile_size_x
                tly = sar_proj_bounds.top - row * tile_size_y
                dst_transform = from_origin(tlx, tly, pixel_size_meters, pixel_size_meters)

                sar_dest = np.zeros((sar_count, tile_height, tile_width), dtype=sar_dtype)
                reproject(
                    source=sar_reproj,
                    destination=sar_dest,
                    src_transform=sar_grid_transform,
                    src_crs=sar_grid_crs,
                    dst_transform=dst_transform,
                    dst_crs=sar_grid_crs,
                    resampling=Resampling.nearest,
                )
                sar_tile_path = os.path.join(sar_out_dir, f"tile_{row}_{col}.tif")
                sar_tile_meta = {
                    'driver': 'GTiff',
                    'height': tile_height,
                    'width': tile_width,
                    'count': sar_count,
                    'dtype': sar_dtype,
                    'crs': sar_grid_crs,
                    'transform': dst_transform,
                    'nodata': sar_nodata
                }
                with rasterio.open(sar_tile_path, 'w', **sar_tile_meta) as dst:
                    dst.write(sar_dest)
                print(f"  Wrote SAR tile: {sar_tile_path}")

                opt_dest = np.zeros((opt_count, tile_height, tile_width), dtype=opt_dtype)
                reproject(
                    source=rasterio.band(opt_src, list(range(1, opt_count + 1))),
                    destination=opt_dest,
                    src_transform=opt_transform,
                    src_crs=opt_crs,
                    dst_transform=dst_transform,
                    dst_crs=sar_grid_crs,
                    resampling=Resampling.nearest,
                )
                opt_tile_path = os.path.join(opt_out_dir, f"tile_{row}_{col}.tif")
                opt_tile_meta = {
                    'driver': 'GTiff',
                    'height': tile_height,
                    'width': tile_width,
                    'count': opt_count,
                    'dtype': opt_dtype,
                    'crs': sar_grid_crs,
                    'transform': dst_transform,
                    'nodata': opt_nodata
                }
                with rasterio.open(opt_tile_path, 'w', **opt_tile_meta) as dst:
                    dst.write(opt_dest)
                print(f"  Wrote optical tile: {opt_tile_path}")

    print("Tiling complete.")
