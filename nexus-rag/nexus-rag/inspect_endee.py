import inspect, endee
src = inspect.getsource(endee.Endee.create_index)
print(src[:3000])
