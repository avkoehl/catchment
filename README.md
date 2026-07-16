# catchment

Given a DEM, channel heads, flow directions, and flow accumulation, extract
the labeled channel network, subbasins, reaches, HAND/REM, and the vectorized
stream network.

```python
from catchment import extract_catchment

stream_network, subbasins, reaches, hand, vector_network = extract_catchment(
    dem, channel_heads, flow_dir, flow_acc
)
```
