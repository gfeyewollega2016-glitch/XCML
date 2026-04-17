// ============================================================================
// Causal GeoXAI – Vegetation Productivity & Fire Disturbance Module
// Yellow River Basin (1990–2025)
//
// PURPOSE:
// This script reconstructs long-term vegetation productivity (NPP),
// disturbance (dNBR), and resilience components (stability, recovery)
// from multi-source Earth observation datasets.
//
// KEY FEATURES:
// - Landsat-derived NDVI (30 m)
// - ERA5-Land climate forcing
// - Light-use efficiency (LUE) NPP model
// - Calibration against MODIS MOD17A3HGF
// - Decadal resilience metrics (stability, recovery rate, recovery ratio)
// - Fire disturbance proxy using dNBR
//
// OUTPUTS:
// - Annual calibrated NPP (1990–2025)
// - Period mean NPP, variability, observation counts
// - Decadal resilience indicators
// - Fire disturbance (dNBR)
//
// NOTE:
// IMPORTANT:
// This module is strictly used for constructing LEREI-X components.
// Outputs are NOT included as covariates in the DML causal model.
// See /causal_model/ for treatment-effect estimation pipeline.
// ============================================================================

var region = roi;
var TARGET_SCALE = 1000;  
var CRS = 'EPSG:4326';
Map.centerObject(region, 6);

// ============================================================================
// 1. CLOUD MASKING (Landsat Collection 2 QA_PIXEL)
// ============================================================================

function maskLandsatC2(img) {
  var qa = img.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(1 << 3).eq(0)   // cloud shadow
    .and(qa.bitwiseAnd(1 << 4).eq(0))      // snow/ice
    .and(qa.bitwiseAnd(1 << 5).eq(0))      // cloud
    .and(qa.bitwiseAnd(1 << 8).eq(0));     // cirrus
  return img.updateMask(mask);
}

function scaleLandsatL2Bands(img, bandList) {
  return img.select(bandList)
    .multiply(0.0000275)
    .add(-0.2)
    .copyProperties(img, ['system:time_start']);
}

function focalGapFill(img, radius) {
  var fill = img.focal_median({radius: radius || 1, units: 'pixels'});
  return img.unmask(fill);
}

// ============================================================================
// 2. LAND / WATER MASK (ESA WorldCover, resampled to 1 km)
// ============================================================================

var landMask = ee.ImageCollection('ESA/WorldCover/v200')
  .first()
  .select('Map')
  .neq(80)
  .rename('land')
  .reproject({crs: CRS, scale: TARGET_SCALE})
  .clip(region);

// ============================================================================
// 3. LANDSAT NDVI (Annual Composites, aggregated to 1 km)
// ============================================================================

var landsatNdvi = ee.ImageCollection('LANDSAT/COMPOSITES/C02/T1_L2_ANNUAL_NDVI')
  .filterBounds(region)
  .select('NDVI')
  .map(function(img) {
    return img
      .reduceResolution({
        reducer: ee.Reducer.mean(),
        bestEffort: true,
        maxPixels: 1e13
      })
      .reproject({crs: CRS, scale: TARGET_SCALE})
      .clip(region);
  });

// ============================================================================
// 4. ERA5-Land Climate Data (reprojected to 1 km)
// ============================================================================

var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
  .filterBounds(region)
  .select(['temperature_2m', 'surface_net_solar_radiation_sum', 'total_precipitation_sum'])
  .map(function(img) {
    return img.reproject({crs: CRS, scale: TARGET_SCALE}).clip(region);
  });

// ============================================================================
// 5. MODIS NPP (Aggregated to 1 km for calibration)
// ============================================================================

var modisNpp = ee.ImageCollection('MODIS/061/MOD17A3HGF')
  .filterBounds(region)
  .filterDate('2001-01-01', '2026-01-01')
  .select('Npp')
  .map(function(img) {
    return img
      .reduceResolution({
        reducer: ee.Reducer.mean(),
        bestEffort: true
      })
      .reproject({crs: CRS, scale: TARGET_SCALE})
      .multiply(0.0001).multiply(1000)
      .rename('NPP')
      .clip(region);
  });

// ============================================================================
// 6. LUE NPP (Landsat-based, MOD17-consistent, at 1 km)
// ============================================================================

var NDVI_MIN = 0.05;
var NDVI_MAX = 0.86;
var EPSILON_MAX = 1.1;
var PAR_FRAC = 0.45;
var WATER_PARAM = 500;

function computeNPP(year) {
  year = ee.Number(year);

  var ndviImg = ee.Image(
    landsatNdvi.filter(ee.Filter.calendarRange(year, year, 'year')).first()
  );

  ndviImg = focalGapFill(ndviImg, 1);

  var clim = era5.filter(ee.Filter.calendarRange(year, year, 'year'));
  var temp = clim.select('temperature_2m').mean().subtract(273.15);
  var rad = clim.select('surface_net_solar_radiation_sum').sum().divide(1e6);
  var prec = clim.select('total_precipitation_sum').sum().multiply(1000);

  var fpar = ndviImg.subtract(NDVI_MIN).divide(NDVI_MAX - NDVI_MIN).clamp(0, 1);
  var fT = temp.clamp(0, 30).divide(30);
  var fW = prec.divide(prec.add(WATER_PARAM)).clamp(0, 1);
  var par = rad.multiply(PAR_FRAC);

  var npp = par.multiply(fpar)
    .multiply(fT)
    .multiply(fW)
    .multiply(EPSILON_MAX)
    .rename('NPP')
    .clip(region)
    .updateMask(landMask);

  return npp.set({
    'year': year,
    'system:time_start': ee.Date.fromYMD(year, 1, 1).millis()
  });
}

var years = ee.List.sequence(1990, 2025);
var nppAnnual = ee.ImageCollection.fromImages(
  years.map(function(y) { return computeNPP(y); })
);

// ============================================================================
// 7. CALIBRATION (LUE vs MODIS, Quadratic Fit)
// ============================================================================

var overlapYears = ee.List.sequence(2001, 2025);

var calibData = ee.FeatureCollection(overlapYears.map(function(y) {
  y = ee.Number(y);

  var lue = ee.Image(nppAnnual.filter(ee.Filter.eq('year', y)).first());
  var mod = ee.Image(modisNpp.filter(ee.Filter.calendarRange(y, y, 'year')).first());

  var lueMean = lue.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: region,
    scale: TARGET_SCALE,
    bestEffort: true,
    maxPixels: 1e13
  }).get('NPP');

  var modMean = mod.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: region,
    scale: TARGET_SCALE,
    bestEffort: true,
    maxPixels: 1e13
  }).get('NPP');

  return ee.Feature(null, {
    year: y,
    LUE: lueMean,
    LUE_sq: ee.Number(lueMean).pow(2),
    MODIS: modMean
  });
}));

// Add constant column for regression
calibData = calibData.map(function(f) {
  return f.set('CONST', 1);
});

// Linear regression: MODIS = a + b*LUE + c*LUE^2
var regress = calibData.reduceColumns({
  reducer: ee.Reducer.linearRegression(3, 1),
  selectors: ['CONST', 'LUE', 'LUE_sq', 'MODIS']
});

var coeffs = ee.Array(regress.get('coefficients')).project([0]);
var a = ee.Number(coeffs.get([0, 0]));
var b = ee.Number(coeffs.get([1, 0]));
var c = ee.Number(coeffs.get([2, 0]));

print('Calibration coefficients (MODIS = a + b*LUE + c*LUE^2):', a, b, c);

// Apply calibration
function calibrate(img) {
  var x = img.select('NPP');
  var cal = ee.Image(a)
    .add(ee.Image(b).multiply(x))
    .add(ee.Image(c).multiply(x.pow(2)))
    .rename('NPP_cal')
    .clamp(0, 3000);
  return img.addBands(cal).copyProperties(img, ['year', 'system:time_start']);
}

var nppCal = nppAnnual.map(calibrate);

// ============================================================================
// 8. MEAN NPP EXPORTS (1 km, Period Averages)
// ============================================================================

var periods = [
  {name: 'NPP_1990_2000', s: 1990, e: 2000},
  {name: 'NPP_2001_2009', s: 2001, e: 2009},
  {name: 'NPP_2010_2020', s: 2010, e: 2020},
  {name: 'NPP_2021_2025', s: 2021, e: 2025}
];

periods.forEach(function(p) {
  var col = nppCal.filter(ee.Filter.calendarRange(p.s, p.e, 'year'));
  var mean = col.select('NPP_cal').mean().rename('NPP_mean');
  var std = col.select('NPP_cal').reduce(ee.Reducer.stdDev()).rename('NPP_std');
  var cnt = col.select('NPP_cal').count().rename('obs_count');

  Export.image.toDrive({
    image: mean.addBands(std).addBands(cnt).toFloat(),
    description: p.name,
    folder: 'YRB_Vegetation',
    fileNamePrefix: p.name,
    region: region,
    scale: TARGET_SCALE,
    crs: CRS,
    maxPixels: 1e13
  });
});

// ============================================================================
// 9. DECADAL STABILITY & RECOVERY METRICS (1 km)
// ============================================================================

var decades = [
  {name: '1990s', b: [1990, 1994], r: [1995, 1999], n: 5},
  {name: '2000s', b: [2000, 2004], r: [2005, 2009], n: 5},
  {name: '2010s', b: [2010, 2014], r: [2015, 2019], n: 5},
  {name: '2020s', b: [2020, 2022], r: [2023, 2025], n: 3}
];

decades.forEach(function(d) {
  var base = nppCal.filter(ee.Filter.calendarRange(d.b[0], d.b[1], 'year'));
  var rec = nppCal.filter(ee.Filter.calendarRange(d.r[0], d.r[1], 'year'));

  var bMean = base.select('NPP_cal').mean();
  var bStd = base.select('NPP_cal').reduce(ee.Reducer.stdDev());
  var rMean = rec.select('NPP_cal').mean();

  var eps = 0.001;
  var stability = bMean.divide(bStd.add(eps)).rename('Stability').clamp(0, 20);
  var rate = rMean.subtract(bMean).divide(d.n).rename('Recovery_Rate').clamp(-100, 100);
  var ratio = rMean.divide(bMean.add(eps)).rename('Recovery_Ratio').clamp(0, 3);

  [
    {img: stability, name: 'Stability'},
    {img: rate, name: 'Recovery_Rate'},
    {img: ratio, name: 'Recovery_Ratio'}
  ].forEach(function(o) {
    Export.image.toDrive({
      image: o.img.toFloat(),
      description: o.name + '_' + d.name,
      folder: 'YRB_Vegetation',
      fileNamePrefix: o.name + '_' + d.name,
      region: region,
      scale: TARGET_SCALE,
      crs: CRS,
      maxPixels: 1e13
    });
  });
});

// ============================================================================
// 10. MODIS MEAN NPP EXPORTS (1 km)
// ============================================================================

var modPeriods = [
  {name: 'MODIS_2001_2009', s: 2001, e: 2009},
  {name: 'MODIS_2010_2020', s: 2010, e: 2020},
  {name: 'MODIS_2021_2025', s: 2021, e: 2025}
];

modPeriods.forEach(function(p) {
  var col = modisNpp.filter(ee.Filter.calendarRange(p.s, p.e, 'year'));
  Export.image.toDrive({
    image: col.mean().rename('NPP').toFloat(),
    description: p.name,
    folder: 'YRB_Vegetation',
    fileNamePrefix: p.name,
    region: region,
    scale: TARGET_SCALE,
    crs: CRS,
    maxPixels: 1e13
  });
});

// ============================================================================
// 11. FIRE DETECTION (dNBR, aggregated to 1 km)
// ============================================================================

var fireYears = [
  {y: 1990, collection: 'LANDSAT/LT05/C02/T1_L2', nir: 'SR_B4', swir: 'SR_B7'},
  {y: 2000, collection: 'LANDSAT/LE07/C02/T1_L2', nir: 'SR_B4', swir: 'SR_B7'},
  {y: 2010, collection: 'LANDSAT/LT05/C02/T1_L2', nir: 'SR_B4', swir: 'SR_B7'},
  {y: 2020, collection: 'LANDSAT/LC08/C02/T1_L2', nir: 'SR_B5', swir: 'SR_B7'},
  {y: 2025, collection: 'LANDSAT/LC09/C02/T1_L2', nir: 'SR_B5', swir: 'SR_B7'}
];

fireYears.forEach(function(cfg) {
  var col = ee.ImageCollection(cfg.collection)
    .filterBounds(region)
    .filter(ee.Filter.lt('CLOUD_COVER', 30))
    .map(maskLandsatC2);

  var scaled = function(img) {
    return scaleLandsatL2Bands(img, [cfg.nir, cfg.swir]);
  };

  var pre = col.filterDate(cfg.y + '-01-01', cfg.y + '-06-30').map(scaled);
  var post = col.filterDate(cfg.y + '-07-01', cfg.y + '-12-31').map(scaled);

  var nbrPre = focalGapFill(pre.median(), 1).normalizedDifference([cfg.nir, cfg.swir]);
  var nbrPost = focalGapFill(post.median(), 1).normalizedDifference([cfg.nir, cfg.swir]);

  var dnbr = nbrPre.subtract(nbrPost).rename('dNBR').clamp(-1, 1);

  // Aggregate to 1 km for consistency 
var dnbr_1km = dnbr
  .reduceResolution({
    reducer: ee.Reducer.mean(),
    maxPixels: 1024
  })
  .reproject({
    crs: CRS,
    scale: TARGET_SCALE
  })
  .updateMask(landMask);

  Export.image.toDrive({
    image: dnbr_1km.toFloat(),
    description: 'dNBR_1km_' + cfg.y,
    folder: 'YRB_Fire',
    fileNamePrefix: 'dNBR_1km_' + cfg.y,
    region: region,
    scale: TARGET_SCALE,
    crs: CRS,
    maxPixels: 1e13
  });
});

// ============================================================================
// 12. EXPORT CALIBRATION TABLE
// ============================================================================

Export.table.toDrive({
  collection: calibData,
  description: 'NPP_Calibration_Table',
  folder: 'YRB_Vegetation',
  fileFormat: 'CSV'
});


print('✅ All exports ready. Check Tasks tab.');
print('📍 Study area centered. Ensure roi is defined.');
