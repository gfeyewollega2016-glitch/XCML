// ============================================================
// TERRAIN ANALYSIS FOR LEREI-X (region of interest roi)
// Sections: 2.3.1 (Resistance: Slope), 2.3.1 (Adaptability: TRI)
// ============================================================

// 1. AREA OF INTEREST
var aoi = roi;  // Uploaded shapefile or defined geometry

// 2. LOAD SRTM DIGITAL ELEVATION MODEL (30m)
var srtm = ee.Image('USGS/SRTMGL1_003')
  .clip(aoi)
  .rename('elevation');

// 3. SLOPE (degrees) – for Resistance component
var slope = ee.Terrain.slope(srtm).rename('slope_deg');

// 4. TERRAIN RUGGEDNESS INDEX (TRI – Riley method) – for Adaptability component
//    Uses 3x3 kernel to compute mean absolute deviation from center pixel
var kernel = ee.Kernel.fixed(3, 3, [
  [1, 1, 1],
  [1, 0, 1],
  [1, 1, 1]
]);
var neighborMean = srtm.reduceNeighborhood({
  reducer: ee.Reducer.mean(),
  kernel: kernel
});
var tri = srtm.subtract(neighborMean).abs()
  .rename('TRI');

// 5. OPTIONAL: Apply land mask (exclude permanent water bodies)
var waterMask = ee.ImageCollection('ESA/WorldCover/v100')
  .filterDate('2020')
  .first()
  .select('Map')
  .neq(80)  // 80 = water class
  .rename('land_mask');

// 6. COMPILE TERRAIN METRICS
var terrain = ee.Image.cat([slope, tri])
  .updateMask(waterMask)
  .clip(aoi);

// 7. VISUALIZATION
Map.centerObject(aoi, 7);
Map.addLayer(slope, {min: 0, max: 30, palette: ['green', 'yellow', 'orange', 'red']}, 'Slope (degrees)');
Map.addLayer(tri, {min: 0, max: 50, palette: ['green', 'yellow', 'red']}, 'TRI (Ruggedness)');

// 8. STATISTICS (optional, for quality check)
var stats = terrain.select(['slope_deg', 'TRI']).reduceRegion({
  reducer: ee.Reducer.mean().combine(ee.Reducer.stdDev(), null, true),
  geometry: aoi,
  scale: 90,
  maxPixels: 1e13,
  bestEffort: true
});
print('Terrain statistics (mean ± stdDev):', stats);

// 9. EXPORT (for downstream Python causal analysis)
Export.image.toDrive({
  image: slope,
  description: 'YRB_Slope_Degrees_30m',
  scale: 30,
  region: aoi,
  maxPixels: 1e13,
  folder: 'GEE_Exports',
  fileFormat: 'GeoTIFF'
});

Export.image.toDrive({
  image: tri,
  description: 'YRB_TRI_Riley_30m',
  scale: 30,
  region: aoi,
  maxPixels: 1e13,
  folder: 'GEE_Exports',
  fileFormat: 'GeoTIFF'
});

Export.image.toDrive({
  image: terrain,
  description: 'YRB_Terrain_Metrics_30m',
  scale: 30,
  region: aoi,
  maxPixels: 1e13,
  folder: 'GEE_Exports',
  fileFormat: 'GeoTIFF'
});

print('Export tasks initiated. Check "Tasks" tab.');


**terrain analysis script**
