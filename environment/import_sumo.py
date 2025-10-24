import os
import sys

def alarm(*msg):
    msg = msg or []
    # ..... bold   blink  red .......................................... reset 
    return "\033[1m\033[5m\033[31m" + " ".join([str(m) for m in msg]) + "\033[0m"

def comment(*msg):
    return "\033[2m\033[32m" + " ".join([str(m) for m in msg]) + "\033[0m"

def path(*args):
    if not args:
        if 'SUMO_HOME' not in os.environ:
            return None
        path = os.path.join(os.environ['SUMO_HOME'], 'tools')
    else:
        path = os.path.join(*args)
    if path[0] != "/":
        path.insert(0, "/")
    if path.endswith("tools"):
        return path
    else:
        return os.path.join(path, "tools")

def check(*args):
    """Return True if was able to install SUMO from a given directory."""
    p = path(*args)
    if p:
        return os.path.isdir(os.path.join(p, "traci"))
    return False


P1 = []
P2 = ["usr", "share", "sumo", "tools"]
P3 = ["opt", "sumo", "tools"]
if check(*P1):
    sys.path.append(path(*P1))
    import traci
    print(comment("Imported from"), P1)
elif check(*P2):
    sys.path.append(path(*P2))
    import traci
    print(comment("IMported from"), P2)
elif check(*P3):
    sys.path.append(path(*P3))
    import traci
    print(comment("Imported from"), P3)
else:
    print(alarm("Could not find SUMO installation w/ tools/!!!"))
    raise ImportError()