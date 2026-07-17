# catchment

Given a raw DEM and channel initiation points (from flowlines or a
contributing-area threshold), extract the full raster description of the
drainage system: conditioned DEM, flow direction/accumulation, labeled
channel network, subbasins, reaches, HAND, and the vectorized stream network.

```python
from catchment import extract_catchment

results = extract_catchment(dem, flowlines=flowlines)
# or: results = extract_catchment(dem, threshold_area=200_000)

results.stream_network
results.subbasins
results.reaches
results.hand
results.vector_network
```
