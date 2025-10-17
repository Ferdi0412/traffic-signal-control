import os
import numpy as np

CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "sumo-networks")
print(CFG_PATH)

### === Helpers ========================================================
def py_index(lst, val):
    try:
        return lst.index(val)
    except ValueError:
        return None

### === Formatting Prints & Type Checking ==============================
COLORS = {'r': 31, 'red': 31, 'g': 32, 'green': 32,
          'y': 33, 'yellow': 33, 'b': 34, 'blue': 34}

def coltxt(msg, col):
    """Apply a text (forgeround) colour to message, and bold."""
    if col is None:
        return msg
    return f"\033[{COLORS[col]}m" + str(msg) + "\033[0m"

def colbg(msg, col):
    """Apply a background to the message."""
    if col is None:
        return msg
    return f"\033[{COLORS[col]+10}m" + msg + "\033[0m"

def alarm(*msg):
    return "\033[1m\033[5m\033[31m" + " ".join([str(m) for m in msg]) + "\033[0m"

def warn(*msg):
    return "\033[1m\033[5m\033[33m" + " ".join([str(m) for m in msg]) + "\033[0m"

def blue(*msg):
    return "\033[1m\033[34m" + " ".join([str(m) for m in msg]) + "\033[0m"

def cyan(*msg):
    return "\033[1m\033[36m" + " ".join([str(m) for m in msg]) + "\033[0m"

def dim(*msg):
    return "\033[2m" + " ".join([str(m) for m in msg]) + "\033[0m"

def comment(*msg):
    return "\033[2m\033[32m" + " ".join([str(m) for m in msg]) + "\033[0m"

def assert_type(target, of_type, *args, subclass=True):
    if not isinstance(of_type, tuple):
        of_type = (of_type,)
    if args:
        of_type = (*of_type, *args)
    if not isinstance(target, type):
        target = type(target)
    valid = issubclass(target, of_type) if subclass else (target in of_type)
    if not valid:
        print(of_type)
        for t in of_type:
            print(t)
        msg = alarm("assert_type") + " Expected one of "
        msg += dim(of_type)
        msg += " but got " + f"<{blue(target.__name__)}>"
        print(msg)
        raise AssertionError()

def notify_error(exception, msg, *args):
    assert_type(exception, BaseException)
    print(alarm(msg), *args)
    if isinstance(exception, type):
        raise exception()
    else:
        raise exception

### === File Path Handling =============================================
def cfg_file(fname, ftype, *, fdir=None):
    if "." in fname:
        fname, *dropped = fname.split(".")
        print(warn("sumocfg"), "Dropped filename following '.':\n",
              comment("." + ".".join([*dropped])))
    return os.path.join(fdir or CFG_PATH, fname + ftype)

### === Simplify Lane Indexing =========================================
def road_index(road):
    """Translate 'N' to 0."""
    if isinstance(road, (int, np.integer)):
        if road not in range(4):
            notify_error(ValueError, "road_index", "Out of bounds", road)
        return road
    if road[0].lower() == 'n':
        return 0
    if road[0].lower() == 'e':
        return 1
    if road[0].lower() == 's':
        return 2
    if road[0].lower() == 'w':
        return 3
    notify_error(ValueError, "road_index:", road)

def turn_index(turn):
    """Translate 'left' to 0."""
    if isinstance(turn, int):
        if turn not in range(3):
            notify_error(ValueError, "turn_index:", "Out of bounds", turn)
        return turn
    if turn[0].lower() == 'l':
        return 0
    elif turn[0].lower() == 'f' or turn[0].lower() == 's':
        return 1
    elif turn[0].lower() == 'r':
        return 2
    notify_error(ValueError, "turn_index:", turn)

def lane_index(road, turn):
    """Translate lane ('n', 'l') to 0."""
    road = road_index(road)
    turn = turn_index(turn)
    return 3 * road + turn

def turns_to(road, turn=None):
    """Translate ('n', 'l') or 0 to lane_out 3."""
    # If turn is None - road is lane, otherwise 'N' or similar
    if turn is None:
        return turns_to(road // 4, turn % 4)
    road = road_index(road)
    turn = turn_index(turn)
    target = road + (turn + 1)
    return target - 4 if target > 3 else target

def turns_from(road, turn=None):
    """Translate lane out ('n', 'l') or 0 to lane_in 9."""
    # If turn is None - road is lane, otherwise 'N' or similar
    if turn is None:
        return turns_from(road // 4, turn % 4)
    road = road_index(road)
    turn = turn_index(turn)
    target = road - (turn + 1)
    return target + 4 if target < 0 else target


### === Geometry =======================================
def ctp(p, p1=None):
    """Cartesian -> pseudo-polar."""
    x, y = p if p1 is None else (p, p1)
    return np.sqrt(x**2 + y**2), np.rad2deg(np.arctan2(y, x))

def ptc(p, p1=None):
    """Pseudo-polar -> cartesian."""
    r, a = p if p1 is None else (p, p1)
    return r * np.sin(np.deg2rad(a)), r * np.cos(np.deg2rad(a))  

def perp(p0, p1, d=None):
    """Line perpendicular to p0->p1, with lentgh d."""
    r, a = ctp(np.array(p1)-np.array(p0))
    return ptc(d or r, a - 90)


def proj(p0, p1, d):
    """Project d from p0 towards p1."""
    p0, p1 = map(np.array, (p0, p1))
    dp = (p1 - p0) / np.linalg.norm(p1 - p0)
    res = p0 + d * dp
    return res[0], res[1]