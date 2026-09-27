"""Minimal netCDF-3 classic reader (no dependencies). Usage: python3 tools/nc3dump.py file.nc [var ...]
Prints dimensions/variables; with variable names, writes them as JSON to stdout."""
import struct, sys, json

def read_nc3(path):
    d = open(path, 'rb').read()
    assert d[:3] == b'CDF', 'not netCDF-3'
    ver = d[3]
    pos = 4
    def u32():
        nonlocal pos; v = struct.unpack('>I', d[pos:pos+4])[0]; pos += 4; return v
    def u64():
        nonlocal pos; v = struct.unpack('>Q', d[pos:pos+8])[0]; pos += 8; return v
    def name():
        nonlocal pos; n = u32(); s = d[pos:pos+n].decode(); pos += (n + 3) // 4 * 4; return s
    sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8}
    fmts = {1: 'b', 2: 'c', 3: 'h', 4: 'i', 5: 'f', 6: 'd'}
    def attrs():
        nonlocal pos
        tag = u32(); n = u32(); out = {}
        for _ in range(n):
            nm = name(); t = u32(); cnt = u32(); sz = sizes[t] * cnt
            raw = d[pos:pos+sz]; pos += (sz + 3) // 4 * 4
            out[nm] = raw.decode(errors='ignore') if t == 2 else list(struct.unpack('>' + fmts[t] * cnt, raw))
        return out
    numrecs = u32()
    tag = u32(); ndims = u32(); dims = []
    for _ in range(ndims):
        dims.append((name(), u32()))
    gatt = attrs()
    tag = u32(); nvars = u32(); vars_ = {}
    for _ in range(nvars):
        nm = name(); nd = u32(); dimids = [u32() for _ in range(nd)]
        at = attrs(); t = u32(); vsize = u32(); begin = u64() if ver == 2 else u32()
        vars_[nm] = dict(dims=[dims[i] for i in dimids], attrs=at, type=t, begin=begin)
    def get(nm):
        v = vars_[nm]; n = 1
        for _, s in v['dims']: n *= s
        t = v['type']
        return list(struct.unpack('>' + fmts[t] * n, d[v['begin']:v['begin'] + sizes[t] * n]))
    return dims, vars_, get

if __name__ == '__main__':
    dims, vars_, get = read_nc3(sys.argv[1])
    if len(sys.argv) == 2:
        print('dims', dims)
        for k, v in vars_.items(): print(k, v['dims'], v['attrs'])
    else:
        json.dump({k: get(k) for k in sys.argv[2:]}, sys.stdout)
