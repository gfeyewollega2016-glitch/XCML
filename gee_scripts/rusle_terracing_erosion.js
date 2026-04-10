
// ============================================================================
// TERRACE-AWARE RUSLE FOR CAUSAL TREATMENT PREPARATION (1 km)
// Climate harmonized with CHIRPS + ERA5-Land
// Output: terrace-induced erosion reduction (continuous treatment)
// ============================================================================

var region = (roi.geometry !== undefined) ? roi.geometry() : roi;
Map.centerObject(region, 6);

var TARGET_SCALE = 1000;
var CRS = 'EPSG:4326';
var ETA = 0.5;     // terrace attenuation strength
var LS_MAX = 20;
var years = [2000, 2010, 2020];

var terracePaths = {
  2000: 'terrace_data_2000',
  2010: 'terrace_data_2010',
  2020: 'terrace_data_2020'
  //.....
};

// ----------------------------------------------------------------------------
// HELPERS
// ----------------------------------------------------------------------------
function to1km(img, method) {
  method = method || 'bilinear';
  return img
    .resample(method)
    .reproject({crs: CRS, scale: TARGET_SCALE})
    .clip(region);
}

function to1kmCategorical(img) {
  // Do not call resample() here; Earth Engine resample() only accepts
  // continuous interpolation modes such as bilinear/bicubic.
  return img
    .reproject({crs: CRS, scale: TARGET_SCALE})
    .clip(region);
}

function loadTerraceFrac(year) {
  return to1km(
    ee.Image(terracePaths[year])
      .select(0)
      .rename('terrace_frac'),
    'bilinear'
  ).clamp(0, 1);
}

function yearStart(year) {
  return ee.Date.fromYMD(year, 1, 1);
}

function yearEnd(year) {
  return ee.Date.fromYMD(year + 1, 1, 1);
}

// ----------------------------------------------------------------------------
// STATIC TERRAIN + SOIL
// ----------------------------------------------------------------------------
var dem = to1km(ee.Image('USGS/SRTMGL1_003').rename('DEM'));
var slopeRad = ee.Terrain.slope(dem).multiply(Math.PI / 180);

var L = ee.Image.constant(1000);
var m = 0.4;
var n = 1.3;

var LSbase = L.divide(22.13).pow(m)
  .multiply(slopeRad.sin().divide(0.0896).pow(n))
  .rename('LS')
  .clamp(0.001, LS_MAX);

var soc = to1km(
  ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02')
    .select('b0')
    .divide(10)
    .rename('SOC_pct')
);

var texture = to1kmCategorical(
  ee.Image('OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02')
    .select('b0')
    .rename('texture_class')
);

var textureWeight = texture.expression(
  "tex==10 || tex==11 ? 0.45 : " +
  "tex==9 ? 0.35 : " +
  "tex==7 || tex==8 ? 0.25 : " +
  "tex<=4 ? 0.18 : 0.22",
  {tex: texture}
).rename('texture_weight');

var socStats = soc.reduceRegion({
  reducer: ee.Reducer.percentile([2, 98]),
  geometry: region,
  scale: TARGET_SCALE,
  bestEffort: true,
  maxPixels: 1e13
});

var socP2 = ee.Number(socStats.get('SOC_pct_p2'));
var socP98 = ee.Number(socStats.get('SOC_pct_p98'));

var socNorm = soc.subtract(socP2)
  .divide(socP98.subtract(socP2))
  .clamp(0, 1);

var K = textureWeight
  .multiply(ee.Image.constant(1).subtract(socNorm.multiply(0.5)))
  .rename('K_factor')
  .clamp(0.05, 0.6);

// ----------------------------------------------------------------------------
// CLIMATE
// ----------------------------------------------------------------------------
function annualPrecipCHIRPS(year) {
  return to1km(
    ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
      .filterBounds(region)
      .filterDate(yearStart(year), yearEnd(year))
      .sum()
      .rename('P_annual_mm')
  );
}

function annualRainDaysERA5(year) {
  return to1km(
    ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
      .filterBounds(region)
      .filterDate(yearStart(year), yearEnd(year))
      .select('total_precipitation_sum')
      .map(function(img) {
        return img.gt(0.01);
      })
      .sum()
      .rename('rain_days')
  );
}

function annualNDVI(year) {
  return to1km(
    ee.ImageCollection('LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL_NDVI')
      .filterBounds(region)
      .filterDate(yearStart(year), yearEnd(year))
      .select('NDVI')
      .mean()
      .rename('NDVI')
  );
}

// ----------------------------------------------------------------------------
// MAIN LOOP
// ----------------------------------------------------------------------------
var summaryList = [];
var visYear = 2020; // default year to visualize
var visReduction;

years.forEach(function(year) {
  var terraceFrac = loadTerraceFrac(year);
  var precip = annualPrecipCHIRPS(year);
  var rainDays = annualRainDaysERA5(year);
  var ndvi = annualNDVI(year);

  var R = precip
    .multiply(rainDays.divide(365).add(0.5))
    .multiply(0.35)
    .rename('R_factor');

  var ndviClamped = ndvi.clamp(-0.99, 0.99);
  var C = ndviClamped
    .divide(ee.Image.constant(1).subtract(ndviClamped))
    .multiply(-2)
    .exp()
    .clamp(0, 1)
    .rename('C_factor')
    .where(ndvi.lte(0), 1);

  // terrace modifies slope length + support practice
  var terraceEffect = terraceFrac.multiply(-ETA).exp();
  var LS_with = LSbase.multiply(terraceEffect);
  var P_with = terraceEffect.rename('P_factor');

  var E_no = R.multiply(K).multiply(LSbase).multiply(C).rename('E_noTerr');
  var E_with = R.multiply(K).multiply(LS_with).multiply(C).multiply(P_with).rename('E_withTerr');

  var reduction = E_no.subtract(E_with)
    .max(0)
    .rename('erosion_reduction');

  var validMask = dem.gt(0);
  reduction = reduction.updateMask(validMask).unmask(0);

  // save one year for map visualization
  if (year === visYear) {
    visReduction = reduction;
  }

  // export raster
  Export.image.toDrive({
    image: reduction.toFloat(),
    description: 'Terrace_Erosion_Reduction_1km_' + year,
    folder: 'YRB_Erosion_1km',
    fileNamePrefix: 'YRB_terrace_erosion_reduction_1km_' + year,
    scale: TARGET_SCALE,
    region: region,
    crs: CRS,
    maxPixels: 1e13
  });

  // safe summary over terraced pixels only
  var terrMask = terraceFrac.gt(0.01);
  var terrReduction = reduction.updateMask(terrMask);

  var countDict = terrReduction.reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: region,
    scale: TARGET_SCALE,
    bestEffort: true,
    maxPixels: 1e13
  });

  var pixelCount = ee.Number(ee.Dictionary(countDict).values().get(0));

  function safeSingleStat(img, reducer) {
    var d = img.reduceRegion({
      reducer: reducer,
      geometry: region,
      scale: TARGET_SCALE,
      bestEffort: true,
      maxPixels: 1e13
    });
    return ee.Number(ee.Dictionary(d).values().get(0));
  }

  var stats = ee.Dictionary(ee.Algorithms.If(
    pixelCount.gt(0),
    ee.Dictionary({
      mean: safeSingleStat(terrReduction, ee.Reducer.mean()),
      stdDev: safeSingleStat(terrReduction, ee.Reducer.stdDev()),
      p50: safeSingleStat(terrReduction, ee.Reducer.percentile([50])),
      p90: safeSingleStat(terrReduction, ee.Reducer.percentile([90])),
      p95: safeSingleStat(terrReduction, ee.Reducer.percentile([95])),
      max: safeSingleStat(terrReduction, ee.Reducer.max())
    }),
    ee.Dictionary({
      mean: null,
      stdDev: null,
      p50: null,
      p90: null,
      p95: null,
      max: null
    })
  ));

  var feature = ee.Feature(null, {
    year: year,
    mean_reduction: stats.get('mean'),
    std_reduction: stats.get('stdDev'),
    median_reduction: stats.get('p50'),
    p90_reduction: stats.get('p90'),
    p95_reduction: stats.get('p95'),
    max_reduction: stats.get('max')
  });

  summaryList.push(feature);

  // print yearly quick summary
  print('Year ' + year + ' reduction summary', feature);

  // print yearly quick summary
  print('Year ' + year + ' reduction summary', feature);
});

var summaryFC = ee.FeatureCollection(summaryList);
print('All-year terrace erosion reduction summary', summaryFC);

Export.table.toDrive({
  collection: summaryFC,
  description: 'Terrace_RUSLE_1km_CHIRPS_ERA5_Summary',
  folder: 'YRB_Erosion_1km',
  fileFormat: 'CSV'
});

// ----------------------------------------------------------------------------
// VISUALIZATION
// ----------------------------------------------------------------------------
var reductionVis = {
  min: 0,
  max: 50,
  palette: ['white', 'yellow', 'orange', 'red', 'darkred']
};

var terraceVis = {
  min: 0,
  max: 1,
  palette: ['white', 'cyan', 'blue']
};

Map.addLayer(loadTerraceFrac(visYear), terraceVis, 'Terrace fraction ' + visYear);
Map.addLayer(visReduction, reductionVis, 'Erosion reduction ' + visYear);
Map.addLayer(LSbase, {min: 0, max: 20, palette: ['white', 'green', 'brown']}, 'LS factor');

print('✅ Clean 1 km terrace-aware RUSLE exports started.');
print('📍 Visualization shown for year: ' + visYear);
print('📦 Outputs: yearly GeoTIFFs + summary CSV');



