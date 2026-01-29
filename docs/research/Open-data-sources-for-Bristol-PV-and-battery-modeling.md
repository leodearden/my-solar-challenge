# Open data sources for Bristol domestic PV + battery modeling

A 100-home solar PV and battery project in Bristol, UK can draw on **extensive open data sources** spanning solar resource assessment, technical performance parameters, electricity markets, housing characteristics, and grid infrastructure. The most valuable resources include PVGIS for solar irradiance (3km resolution, hourly data from 2005-2020), the EPC database for Bristol housing characteristics (**25+ million domestic records** with roof descriptions), Elexon's BMRS for wholesale electricity prices (free API, no authentication required), and the Carbon Intensity API for half-hourly regional grid data including the South West region covering Bristol.

This guide identifies **50+ open datasets** with API access, licensing details, and specific Bristol applicability across all eleven requested categories.

## Solar irradiance and weather data provide high-resolution Bristol coverage

**PVGIS (Photovoltaic Geographical Information System)** from the EU Joint Research Centre stands out as the primary resource for solar modeling. It provides hourly Global Horizontal Irradiance (GHI), Direct Normal Irradiance (DNI), and Diffuse Horizontal Irradiance (DHI) at **3km spatial resolution** via the SARAH2 satellite database (2005-2020) or 28km resolution via ERA5 (1985-present). The free REST API requires no authentication and delivers Typical Meteorological Year (TMY) datasets in CSV, JSON, or EnergyPlus Weather (EPW) formats. For Bristol coordinates (51.45°N, 2.58°W), TMY data can be retrieved directly:

```
https://re.jrc.ec.europa.eu/api/v5_3/tmy?lat=51.45&lon=-2.58&outputformat=json
```

**Met Office MIDAS Open** via CEDA Archive provides ground-truth hourly solar radiation measurements from UK stations dating back to 1947, available under the Open Government Licence. Registration is free at CEDA, with data in BADC-CSV format. Bristol-area measurements come from South West England stations. For historical weather parameters (temperature, cloud cover, humidity), **HadUK-Grid** offers 1km gridded daily data from 1836-present in NetCDF format.

**ERA5 reanalysis** from Copernicus provides the longest continuous hourly record (1940-present) at 31km resolution with comprehensive atmospheric variables including solar radiation components. Data is freely accessible via the Copernicus Climate Data Store after registration. For real-time forecasting needs, **Open-Meteo** offers a free API requiring no authentication, combining multiple NWP models at 7km resolution for the UK.

| Source | Resolution | Coverage | Solar Variables | Access |
|--------|------------|----------|-----------------|--------|
| PVGIS SARAH2 | 3km, hourly | 2005-2020 | GHI, DNI, DHI, TMY | Free API |
| NASA POWER | 50-100km, hourly | 2001-present | GHI, DNI, DHI | Free API |
| ERA5 | 31km, hourly | 1940-present | SSRD components | Free (CDS account) |
| MIDAS Open | Station-based, hourly | 1947-2023 | GHI, DNI (limited) | Free (CEDA account) |

## PV degradation rates are well-characterized by NREL and IEA studies

**NREL's comprehensive meta-analysis** (Publication NREL/JA-5200-51664) analyzing 2,000+ degradation measurements over 40 years established the industry-standard parameters: **median 0.5%/year** for crystalline silicon modules, with 78% of systems showing degradation below 1%/year. The 2024 NREL fleet study of 8GW confirmed these rates, finding **0.75%/year median** performance loss across 6,400+ systems.

For UK-specific conditions, the temperate climate provides favorable operating temperatures that typically result in degradation at or below global medians. Northern European studies have recorded rates as low as 0.17%/year over 25 years in optimal conditions. The **IEA-PVPS Task 13** reports (latest February 2025) document degradation mechanisms including cell cracking (up to 7%/year in severe cases) and Potential Induced Degradation (PID) concerns for newer TOPCon and SHJ technologies.

**Recommended modeling parameters for UK conditions:**

| Parameter | Conservative | Typical | Optimistic |
|-----------|--------------|---------|------------|
| Annual degradation | 0.7%/year | 0.5%/year | 0.4%/year |
| 25-year retained capacity | 83% | 88% | 90% |
| First-year degradation | 2-3% | 1-2% | 0.5-1% |

## Battery performance data from independent testing centres

The **Australian ITP Battery Test Centre** (ARENA-funded) provides the most comprehensive independent testing data for domestic batteries, published in freely downloadable reports. Testing of major products under accelerated 3-cycle-per-day protocols revealed:

**Round-trip efficiency (DC):** LiFePO4 batteries (Pylontech, Sony Fortelion) achieve **~95%** efficiency, while NMC systems (Tesla Powerwall 2, LG Chem) typically achieve **89-91%** AC-coupled. GivEnergy reports **92-92.5%** for their hybrid DC-coupled LFP systems.

**Cycle life data from ITP testing:**
- Sony Fortelion LFP: **81% State of Health after 3,680 cycles** (projected 7,390 cycles to 60% SOH)
- Pylontech US2000B LFP: 77% SOH after 2,830 cycles
- Tesla Powerwall 2 NMC: 79% SOH after 2,520 cycles
- BYD B-Box LVS LFP: **93% SOH after 1,060 cycles** (exceptional early performance)

Manufacturer warranties provide minimum performance guarantees: Tesla offers 10 years with 70% capacity retention (unlimited cycles), while GivEnergy warrants 10-12 years at 80% capacity.

| Chemistry | Round-trip efficiency | Cycles to 80% SOH | Calendar life |
|-----------|----------------------|-------------------|---------------|
| LiFePO4 (LFP) | 92-95% | 4,000-6,000 | 15-20 years |
| NMC | 88-90% | 2,500-4,000 | 10-15 years |

## Electricity pricing data accessible via free Elexon and Octopus APIs

**Elexon's BMRS/Insights Solution** provides the authoritative source for GB wholesale electricity prices with **no API key required**. Half-hourly settlement period data includes System Prices, Market Index Data (day-ahead auction reference prices), and generation by fuel type. Historical data extends from 2001 with near-real-time updates.

```
https://data.elexon.co.uk/bmrs/api/v1/datasets/MID?from=2024-01-01&to=2024-01-02
```

**Octopus Energy's public API** enables tracking of dynamic tariffs critical for battery optimization. Agile Octopus prices (half-hourly, day-ahead) and Octopus Go rates are freely accessible:

```
https://api.octopus.energy/v1/products/AGILE-FLEX-22-11-25/electricity-tariffs/E-1R-AGILE-FLEX-22-11-25-A/standard-unit-rates/
```

**OFGEM price cap data** provides quarterly retail tariff benchmarks since January 2019, downloadable as Excel files under Open Government Licence. The cap includes component breakdowns (wholesale, network, policy costs).

For projections, **NESO Future Energy Scenarios** (annual publication, typically July) provides pathways to 2050 with demand, generation mix, and price scenarios. **DESNZ Energy and Emissions Projections** offer government forecasts updated annually.

## Smart Export Guarantee rates require manual tracking

No centralised API exists for SEG rates, which vary significantly by supplier and can change without notice. Current rates (December 2024) range from **4.1p/kWh** (Octopus SEG for non-customers) to **30.31p/kWh** (Octopus Intelligent Flux peak periods for qualifying battery systems).

**Key current rates:**
- Octopus Outgoing Fixed: 15.00p/kWh
- British Gas Export & Earn Flex: 15.1p/kWh
- Good Energy Solar Savings Exclusive: 25.00p/kWh (installer requirement)
- Scottish Power SmartGen Premium: 12.00p/kWh

**OFGEM's SEG Annual Report** provides aggregate statistics: SEG Year 4 (Apr 2023 - Mar 2024) recorded **283.1 GWh total exports** with £30.7 million in payments, 99.98% from solar PV. The **Solar Energy UK SEG League Table** offers regularly updated rate comparisons.

## Interest rate data from Bank of England extends back centuries

The **Bank of England Statistical Interactive Database** provides comprehensive interest rate data under Open Government Licence:

- **Bank Rate history** from 1694 to present (direct Excel download available)
- Quoted household mortgage and loan rates
- Effective rates on outstanding credit

For solar financing models, the relevant series include Bank Rate (base rate) and quoted household secured/unsecured lending rates, all accessible via the database at no cost without registration.

## Bristol housing stock data combines EPC records with LiDAR roof analysis

**The EPC Open Data portal** (epc.opendatacommunities.org) provides the most comprehensive property-level dataset for Bristol modeling. Filter by LOCAL_AUTHORITY = "E06000023" or postcode prefix "BS" to access:

- **ROOF_DESCRIPTION**: Pitched/flat, insulation status
- **PROPERTY_TYPE**: Detached, semi-detached, terraced, flat
- **TOTAL_FLOOR_AREA**: Property size in m²
- **PHOTOVOLTAIC_SUPPLY**: Existing PV installations (% of roof area)
- **UPRN**: Enables linkage to other datasets
- **CONSTRUCTION_AGE_BAND**: Building period

The database contains **25+ million domestic records** since October 2008, updated monthly. Registration is free, with bulk downloads (~5.6GB for all England & Wales) or filtered local authority extracts.

**Environment Agency LiDAR Composite data** enables roof pitch, orientation, and area analysis. The 1m resolution Digital Surface Model (DSM) covers >60% of England, with Bristol well-represented. Download free GeoTiff tiles (5km OS grid squares) from environment.data.gov.uk under Open Government Licence.

**OS OpenMap Local** provides building footprints for Bristol under OGL, updated every 6 months in GeoPackage or Shapefile format via the OS Data Hub (free registration).

**Recommended integration approach:** Link OS building footprints → EPC data via UPRN → LiDAR analysis for roof geometry → INSPIRE Index Polygons for property boundaries.

## Domestic consumption patterns available from multiple academic datasets

**The REFIT dataset** from the University of Strathclyde provides the highest-resolution UK consumption data: **20 households monitored for 2 years** at 8-second intervals, capturing whole-house aggregate power plus 9 individual appliances. Freely downloadable under Creative Commons licence with 1.2 billion readings.

**UK Power Networks' Low Carbon London dataset** offers half-hourly data from **5,567 London households** (2011-2014), including 1,100 on dynamic time-of-use tariffs. Available via London Datastore with registration.

**Elexon Load Profiles** provide GB-wide domestic consumption patterns for Profile Classes 1 (unrestricted) and 2 (Economy 7), with half-hourly coefficients accounting for seasonal and day-type variations.

The **NESO Data Portal** provides system-wide demand data via free CKAN-based API access.

## Grid carbon intensity API provides regional South West data

The **Carbon Intensity API** (carbonintensity.org.uk) delivers half-hourly actual and forecast carbon intensity with **no authentication required**:

- National and **14 DNO regional** breakdowns (South West region covers Bristol)
- Generation mix by fuel type
- **96+ hour forecasts**
- Historical data from September 2017

```
https://api.carbonintensity.org.uk/regional/regionid/13  (South West region)
```

Data is CC BY 4.0 licensed, developed by NESO with Environmental Defense Fund Europe, Oxford University, and WWF.

## DNO network data from National Grid Electricity Distribution

**NGED Connected Data Portal** (connecteddata.nationalgrid.co.uk) provides South West licence area data including Bristol. Key datasets:

- **BSP Power Flow Data**: 5-minute interval power flows by Bulk Supply Point
- **LCT Connections**: Low carbon technology connections by primary substation (from April 2017)
- **Long Term Development Statement**: Network capacity and connection opportunities
- **Electric Nation Dataset**: 2 million hours of EV charging data from trials

API access requires authentication token (from June 2024). Registration is free for most datasets.

## Heat pump and EV adoption data inform load profile projections

**MCS Data Dashboard** (mcscertified.com) tracks every certified installation since 2008/2009 at local authority level: **275,000+ heat pump installations** and **1.5 million+ solar PV** cumulative, with daily updates. Free registration required.

**DfT Vehicle Licensing Statistics** (table VEH0142) provide quarterly EV registrations by local authority, enabling Bristol-specific analysis. **Zap-Map** provides charge point statistics (87,168 UK public points as of November 2025) with free monthly updates.

## Bristol-specific resources consolidate local data

**Open Data Bristol** (opendata.bristol.gov.uk) hosts 140+ datasets including council housing locations, transport data, and planning applications under Open Government Licence.

**Bristol One City Climate Strategy** and **City Leap** partnership documents provide local emissions baselines and decarbonisation targets. The **Impact Community Carbon Calculator** (Centre for Sustainable Energy/Exeter University) offers ward-level carbon footprints.

## Conclusion

Bristol PV+battery modeling benefits from **mature UK open data infrastructure** with comprehensive coverage across all requested categories. The strongest resources are:

1. **PVGIS** for solar resource assessment (3km satellite data, free API)
2. **EPC database** for housing characteristics (25M+ records, UPRN-linkable)
3. **Elexon BMRS** for wholesale prices (free, no-auth API)
4. **Octopus API** for dynamic tariff data (public endpoints)
5. **Carbon Intensity API** for grid optimization (regional forecasts)
6. **NGED Connected Data** for network constraints (South West coverage)
7. **Environment Agency LiDAR** for roof analysis (1m resolution)

All primary sources operate under Open Government Licence, Creative Commons, or equivalent permissive terms enabling commercial and research use with attribution. API access is available for most critical datasets, supporting automated model updates and real-time optimization.