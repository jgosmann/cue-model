from collections import deque, defaultdict, namedtuple


DataRep = namedtuple('DataRep', ['format', 'data'])


registry = defaultdict(dict)


def register_conversion(from_format, to_format):
    def _register(fn):
        registry[from_format][to_format] = fn
        return fn
    return _register


def find_conversion(from_format, to_format):
    to_process = deque([(from_format, [])])
    processed = set()
    while len(to_process) > 0:
        current = to_process.popleft()
        if current[0] == to_format:
            return current[1]
        if current[0] not in processed:
            processed.add(current[0])
            to_process.extend(
                (fmt, current[1] + [fn])
                for fmt, fn in registry[current[0]].items())
    raise RuntimeError(
        "No conversion from {} to {}.".format(from_format, to_format))


def convert(data_rep, to_format):
    from_format = data_rep.format
    data = data_rep.data
    for fn in find_conversion(from_format, to_format):
        data = fn(data)
    return DataRep(to_format, data)
