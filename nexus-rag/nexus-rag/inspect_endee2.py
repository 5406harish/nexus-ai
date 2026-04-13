import inspect, endee
from endee.endee import CHECKSUM
print("CHECKSUM:", CHECKSUM)

# Also check get_index source
src = inspect.getsource(endee.Endee.get_index)
print(src[:2000])
