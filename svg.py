import math
import re
import string
import sys
import xml.etree.ElementTree as etree
from copy import deepcopy
from random import random


class matrix:
    @staticmethod
    def zero(m, n):
        # Create zero matrix
        new_matrix = [[0 for row in range(n)] for col in range(m)]
        return new_matrix

    @staticmethod
    def rand(m, n):
        # Create random matrix
        new_matrix = [[random.random() for row in range(n)] for col in range(m)]
        return new_matrix

    @staticmethod
    def show(matrix):
        # Print out matrix
        for col in matrix:
            print(col)

    @staticmethod
    def mult(matrix1, matrix2):
        # Matrix multiplication
        if len(matrix1[0]) != len(matrix2):
            # Check matrix dimensions
            print('Matrices must be m*n and n*p to multiply!')
        else:
            # Multiply if correct dimensions
            new_matrix = matrix.zero(len(matrix1), len(matrix2[0]))
            for i in range(len(matrix1)):
                for j in range(len(matrix2[0])):
                    for k in range(len(matrix2)):
                        new_matrix[i][j] += matrix1[i][k] * matrix2[k][j]
            return new_matrix


def normalize(s):
    """Normalize a string corresponding to an array of various values."""
    s = s.replace('E', 'e')
    s = re.sub('(?<!e)-', ' -', s)
    s = re.sub('[ \n\r\t,]+', ' ', s)
    s = re.sub(r'(\.[0-9-]+)(?=\.)', r'\1 ', s)
    return s.strip()


UNITS = {
    'mm': 1,
    'cm': 10,
    'in': 2.54,
    'pt': 1 / 72.,
    'pc': 1 / 6.,
    'px': None,
}


def size(string):
    if not string:
        return 0
    try:
        return float(string)
    except ValueError:
        # Not a float, try something else
        pass

    string = normalize(string).split(' ', 1)[0]

    for unit, coefficient in UNITS.items():
        if string.endswith(unit):
            number = float(string[:-len(unit)])
            return number * (coefficient if coefficient else 1)

    # Unknown size
    return 0


class Arc(object):
    def __init__(self, start, radius, rotation, arc, sweep, end):
        """radius is complex, rotation is in degrees,
           large and sweep are 1 or 0 (True/False also work)"""

        self.start = start
        self.radius = radius
        self.rotation = rotation
        self.arc = bool(arc)
        self.sweep = bool(sweep)
        self.end = end

        self._parameterize()

    def __repr__(self):
        return "Arc(start=%s, radius=%s, rotation=%s, arc=%s, sweep=%s, end=%s, tetha=%s, delta=%s)" % (
            self.start,
            self.radius,
            self.rotation,
            self.arc,
            self.sweep,
            self.end,
            self.theta,
            self.delta
        )

    def _parameterize(self):
        # Conversion from endpoint to center parameterization
        # http://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
        if self.start == self.end:
            # This is equivalent of omitting the segment, so do nothing
            return

        if self.radius.real == 0 or self.radius.imag == 0:
            # This should be treated as a straight line
            return

        cosr = math.cos(math.radians(self.rotation))
        sinr = math.sin(math.radians(self.rotation))
        dx = (self.start.real - self.end.real) / 2
        dy = (self.start.imag - self.end.imag) / 2
        x1prim = cosr * dx + sinr * dy
        x1prim_sq = x1prim * x1prim
        y1prim = -sinr * dx + cosr * dy
        y1prim_sq = y1prim * y1prim

        rx = self.radius.real
        rx_sq = rx * rx
        ry = self.radius.imag
        ry_sq = ry * ry

        # Correct out of range radii
        radius_scale = (x1prim_sq / rx_sq) + (y1prim_sq / ry_sq)
        if radius_scale > 1:
            radius_scale = math.sqrt(radius_scale)
            rx *= radius_scale
            ry *= radius_scale
            rx_sq = rx * rx
            ry_sq = ry * ry
            self.radius_scale = radius_scale
        else:
            # SVG spec only scales UP
            self.radius_scale = 1

        t1 = rx_sq * y1prim_sq
        t2 = ry_sq * x1prim_sq
        c = math.sqrt(abs((rx_sq * ry_sq - t1 - t2) / (t1 + t2)))

        if self.arc == self.sweep:
            c = -c
        cxprim = c * rx * y1prim / ry
        cyprim = -c * ry * x1prim / rx

        self.center = complex(
            (cosr * cxprim - sinr * cyprim) + ((self.start.real + self.end.real) / 2),
            (sinr * cxprim + cosr * cyprim) + ((self.start.imag + self.end.imag) / 2),
        )

        ux = (x1prim - cxprim) / rx
        uy = (y1prim - cyprim) / ry
        vx = (-x1prim - cxprim) / rx
        vy = (-y1prim - cyprim) / ry
        n = math.sqrt(ux * ux + uy * uy)
        p = ux
        theta = math.degrees(math.acos(p / n))
        if uy < 0:
            theta = -theta
        self.theta = theta % 360

        n = math.sqrt((ux * ux + uy * uy) * (vx * vx + vy * vy))
        p = ux * vx + uy * vy
        d = p / n
        # In certain cases the above calculation can through inaccuracies
        # become just slightly out of range, f ex -1.0000000000000002.
        if d > 1.0:
            d = 1.0
        elif d < -1.0:
            d = -1.0
        delta = math.degrees(math.acos(d))
        if (ux * vy - uy * vx) < 0:
            delta = -delta
        self.delta = delta % 360
        if not self.sweep:
            self.delta -= 360

    def point(self, pos):
        if self.start == self.end:
            # This is equivalent of omitting the segment
            return self.start

        if self.radius.real == 0 or self.radius.imag == 0:
            # This should be treated as a straight line
            distance = self.end - self.start
            return self.start + distance * pos

        angle = math.radians(self.theta + (self.delta * pos))
        cosr = math.cos(math.radians(self.rotation))
        sinr = math.sin(math.radians(self.rotation))
        radius = self.radius * self.radius_scale

        x = (
            cosr * math.cos(angle) * radius.real
            - sinr * math.sin(angle) * radius.imag
            + self.center.real
        )
        y = (
            sinr * math.cos(angle) * radius.real
            + cosr * math.sin(angle) * radius.imag
            + self.center.imag
        )
        return complex(x, y)


class GC:
    """
    Plotting Context
    each context has a transformation matrix m
    https://www.w3.org/TR/SVG11/coords.html
    """
    # http://www.w3.org/TR/SVG/coords.html#TransformMatrixDefined
    def __init__(self):
        self.a = 1
        self.b = 0
        self.c = 0
        self.d = 1
        self.e = 0
        self.f = 0
        self.m = [[self.a, self.c, self.e], [self.b, self.d, self.f], [0.0, 0.0, 1.0]]
        self.vxy = []   # xy accumulator vector
        self.vvxy = [self.vxy]  # vector of vectors
        self.cgv = []

    def matrix(self, a, b, c, d, e, f):
        tm = [[a, c, e], [b, d, f], [0.0, 0.0, 1.0]]
        self.m = matrix.mult(self.m, tm)

    def translate(self, tx, ty):
        # or [1 0 0 1 tx ty], where tx and ty are the distances to translate coordinates in X and Y, respectively.
        tm = [[1, 0, tx], [0, 1, ty], [0.0, 0.0, 1.0]]
        self.m = matrix.mult(self.m, tm)

    def scale(self, sx, sy):
        # or [sx 0 0 sy 0 0]. One unit in the X and Y directions in the new coordinate system equals sx and sy units in the previous coordinate
        tm = [[sx, 0, 0], [0, sy, 0], [0, 0, 1]]
        self.m = matrix.mult(self.m, tm)

    def rotate(self, a):
        # rotation
        tm = [[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]
        self.m = matrix.mult(self.m, tm)

    def skewX(self, a):
        tm = [[1, math.tan(a), 0], [0, 1, 0], [0, 0, 1]]
        self.m = matrix.mult(self.m, tm)

    def skewY(self, a):
        tm = [[1, 0, 0], [math.tan(a), 1, 0], [0, 0, 1]]
        self.m = matrix.mult(self.m, tm)

    def transform(self, xy):
        tm = [[xy[0]], [xy[1]], [1]]
        tmx = matrix.mult(self.m, tm)
        return tmx[0][0], tmx[1][0]

    def addtransforms(self, transform):
        # print ("addtransformation")
        transformations = transform.split(")")
        for transformation in transformations:
            for ttype in (
                    "scale", "translate", "matrix", "rotate", "skewX",
                    "skewY"):
                if ttype in transformation:
                    transformation = transformation.replace(ttype, "")
                    transformation = transformation.replace("(", "")
                    transformation = normalize(transformation).strip() + " "
                    values = []
                    while transformation:
                        value, transformation = transformation.split(" ", 1)
                        # TODO: manage the x/y sizes here
                        values.append(value)
                    if ttype == 'translate':
                        self.translate(float(values[0]), float(values[1]))
                    elif ttype == 'scale':
                        self.scale((float(values[0])), float(values[1]))
                    elif ttype == 'rotate':
                        self.rotate(math.radians(float(values[0])))
                    elif ttype == 'skewX':
                        self.skewX(math.radians(float(values[0])))
                    elif ttype == 'skewY':
                        self.skewY(math.radians(float(values[0])))
                    elif ttype == 'matrix':
                        # print ("matrix transformation")
                        a, b, c, d = convertToFloats(values[:4])
                        e, f = convertToFloats(values[4:])
                        self.matrix(a, b, c, d, e, f)
                    else:
                        pass

    def plot(self, xy):
        # print (xy)
        self.vxy.append(xy)

    def comment(self, comment):
        self.vxy.append(comment)

    def addchild(self, gc):
        self.vxy.append(gc)

    def newvxy(self):
        self.vxy = []
        self.vvxy.append(self.vxy)

    def applytransform(self):
        outv = []
        for v in self.vvxy:
            outv.append("[new-path]")
            for i in range(len(v)):
                if isinstance(v[i], tuple):
                    #print("before {}".format(v[i]))
                    outv.append(self.transform(v[i]))
                    #print("after {}".format(self.transform(v[i])))
                elif isinstance(v[i], GC):
                    vout = v[i].applytransform()
                    vout2 = []
                    for v2 in vout:
                        if isinstance(v2, tuple):
                            vout2.append(self.transform(v2))
                        elif isinstance(v2, GC):
                            print ("could not happen")
                            pass
                        else:
                            vout2.append(v2)
                    outv += vout2
                else:
                    outv.append(v[i])
        return outv



def getCoeff(x):
    line = [1]
    for i in range(x):
        line = [0] + line
        for j in range(len(line) - 1):
            line[j] += line[j + 1]
    return line


def bez(delta, data):
    """ compute Bezier point """
    coeff = getCoeff(len(data) - 1)
    exp = [(i, len(data) - (i + 1)) for i in range(len(data))]
    t = 0
    points = []
    while t <= 1:
        x = 0
        y = 0
        for i in range(len(data)):
            x += coeff[i] * data[i][0] * (t ** (exp[i][0])) * ((1 - t) ** (exp[i][1]))
            y += coeff[i] * data[i][1] * (t ** (exp[i][0])) * ((1 - t) ** (exp[i][1]))
        points.append((x, y))
        t += delta

    points.append((data[-1][0], data[-1][1]))
    return points


def convertToFloats(alist):
    """Convert number strings in list to floats (leave rest untouched)."""
    for i in range(len(alist)):
        try:
            alist[i] = float(alist[i])
        except ValueError:
            try:
                alist[i] = alist[i].encode("ASCII")
            except:
                pass

    return alist


def convert_quadratic_to_cubic_path(q0, q1, q2):
    """
    Convert a quadratic Bezier curve through q0, q1, q2 to a cubic one.
    """
    c0 = q0
    c1 = (q0[0] + 2. / 3 * (q1[0] - q0[0]), q0[1] + 2. / 3 * (q1[1] - q0[1]))
    c2 = (c1[0] + 1. / 3 * (q2[0] - q0[0]), c1[1] + 1. / 3 * (q2[1] - q0[1]))
    c3 = q2
    return c0, c1, c2, c3


def normaliseSvgPath(attr):
    # operator codes mapped to the minimum number of expected arguments 
    ops = {'A': 7, 'a': 7,
           'Q': 4, 'q': 4, 'T': 2, 't': 2, 'S': 4, 's': 4,
           'M': 2, 'L': 2, 'm': 2, 'l': 2, 'H': 1, 'V': 1,
           'h': 1, 'v': 1, 'C': 6, 'c': 6, 'Z': 0, 'z': 0}

    # do some preprocessing
    opKeys = ops.keys()
    a = attr
    a = a.replace(',', ' ')

    a = a.replace('e-', 'ee')
    a = a.replace('-', ' -')
    a = a.replace('ee', 'e-')
    for op in opKeys:
        a = a.replace(op, " %s " % op)
    a = a.strip()
    a = a.split()

    # insert op codes for each argument of an op with multiple arguments
    res = []
    i = 0
    while i < len(a):
        el = a[i]
        if el in opKeys:
            lastel = el
            res.append(el)
            if el in ('z', 'Z'):
                res.append([])
            else:
                res.append(a[i + 1:i + 1 + ops[el]])
                i = i + ops[el]
            i = i + 1
        else:
            while i < len(a):
                if a[i] not in opKeys:
                    el = lastel
                    if lastel in ('m', 'l'): el = 'l'
                    res.append(el)
                    res.append(a[i:i + ops[el]])
                    i = i + ops[el]
                else:
                    break

    return res




def doPath(path, gc):
    """ do a path"""

#    gc = GC()
    # px,py = current point
    px = 0
    py = 0
    sx = None
    sy = None
    lastop = '?'
    dd = path
    first = True
    normPath = normaliseSvgPath(dd)
    for i in range(0, len(normPath), 2):
        op, nums = normPath[i:i + 2]
        gc.comment("{},{}".format(op, nums))
        relative = op.islower()  # True =relative
        absolute = not relative
        op = op.lower()
# start a new subpath
# (x y)+
        if op == 'm':
            x, y = convertToFloats(nums)
            if first or absolute:
                px = x; py = y
                first = False
            else:
                px = px+x
                py = py+y
            if sx is None:
                sx = px
                sy = py

            if len(gc.vxy):
                gc.newvxy()
            gc.plot((px, py))
# line
# (x y)+
        elif op == 'l':
            x, y = convertToFloats(nums)
            px = x if absolute else px+x
            py = y if absolute else py+y
            gc.plot((px, py))
# vertical line
# y+
        elif op == 'v':
            y = convertToFloats(nums)[0]
            py = y if absolute else py+y
            gc.plot((px, py))

# horizontal line
# x+
        elif op == 'h':
            x = convertToFloats(nums)[0]
            px = x if absolute else px+x
            gc.plot((px, py))

        elif op == 'c':
            """ cubic bezier """
            x1, y1, x2, y2, x, y = convertToFloats(nums)
            spx = px
            spy = py
            if relative: x1 += spx
            if relative: x2 += spx
            if relative: y1 += spy
            if relative: y2 += spy
            px = px + x if relative else x
            py = py + y if relative else y
            pts = bez(0.05, [(spx, spy), (x1, y1), (x2, y2), (px, py)])
            for pt in pts:
                gc.plot(pt)
            # gc.plot((px, py))
            px2 = x2
            py2 = y2

        elif op == 's':
            x2, y2, x, y = convertToFloats(nums)
            # todo
            spx = px
            spy = py
            if relative: x2 += spx
            if relative: y2 += spy

            px = px + x if relative else x
            py = py + y if relative else y

            if lastop in "cs":
                x1 = spx * 2 - px2
                y1 = spy * 2 - py2
            else:
                x1, y1 = spx, spy
            pts = bez(0.05, [(spx, spy), (x1, y1), (x2, y2), (px, py)])
            for pt in pts:
                gc.plot(pt)
            px2 = x2
            py2 = y2


        elif op == 'q':
            """ quadratic bezier """
            x1, y1, x, y = convertToFloats(nums)
            spx = px
            spy = py
            if relative: x1 += spx
            if relative: y1 += spy
            px = px + x if relative else x
            py = py + y if relative else y
            pts = bez(0.05, convert_quadratic_to_cubic_path((spx, spy), (x1, y1),(px, py)))
            for pt in pts:
                gc.plot(pt)
        elif op == 't':
            """smooth curve"""
            x, y = convertToFloats(nums)
            spx = px
            spy = py
            px = px + x if relative else x
            py = py + y if relative else y
            if lastop in "qt":
                x1 = spx + spx - x1
                y1 = spy + spy - y1
            else:
                x1, y1 = spx, spy
            pts = bez(0.05, convert_quadratic_to_cubic_path((spx, spy), (x1, y1),(px, py)))
            for pt in pts:
                gc.plot(pt)

        elif op == 'a':
            """ elliptical arc (rx ry x-axis-rotation large-arc-flag sweep-flag x y) """
            # http://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
            x1 = px
            y1 = py
            rx, ry, xrot, large, sweep, x2, y2 = convertToFloats(nums)
            #print ("Arc radii %s,%s x-axis-rotation %s large-arc %s sweep-flag %s x %s y %s" % (rx,ry,xrot,large,sweep,x2,y2))
            rx, ry, x2, y2 = convertToFloats([rx, ry, x2, y2])
            # convert x2, y2 from relative to absolute
            large = bool(large);
            sweep = bool(sweep)
            if relative: x2 += px
            if relative: y2 += py
            px = x2; py = y2

            arc = Arc(complex(x1, y1), complex(rx, ry), xrot, large, sweep, complex(x2, y2))
            a = 0
            while a < 1:
                arcp = arc.point(a)
                gc.plot((arcp.real, arcp.imag))
                a=a+0.01
            gc.plot((px, py))
        elif op == 'z':
# close the current subpath connecting to initial point
            px=sx; py=sy
            gc.plot((sx, sy))
            if len(gc.vxy):
                gc.newvxy()
            sx = None
        else:
            pass
        lastop = op

    if len(gc.vxy):
        gc.newvxy()

 #   return gc


recursion = 0
def print_recursion (str):
    s = ""
    for i in range(recursion):
        s = s+"\t"
    print(s+str)


def doElements(elements):
    global recursion

    recursion += 1
    gcm = GC()
    for entry in elements:
        gc = GC()
        gcm.addchild(gc)
        # do element
        node_id = entry.get('id')
        if node_id is None:
            node_id = "unspecified"
        print_recursion("rendering id \"{}\"".format(node_id))
        transform = entry.get('transform')
        if transform is not None:
            print_recursion("... with transform {} :{}".format(node_id, transform))
            gc.addtransforms(transform)

        childrens = list(entry)
        print_recursion ("childrens :{}".format(len(childrens)))
        if len(childrens):
            gc.addchild(doElements(entry))

        if entry.tag == '{http://www.w3.org/2000/svg}path':
            d = entry.get('d')
            if d:
                #print_recursion("rendering path id \"{}\"".format(node_id))
                gc.comment("path id: {}".format(node_id))
                doPath(d, gc)

        # all elements done

    recursion -= 1
    return gcm

class SVG:
    def __init__(self, gc, width, height):
        self.gc = gc
        self.width = width
        self.height = height


def doSVG(s, xpath):

    tree = etree.parse(s)
    svgtree = tree.getroot()

    width = svgtree.get("width")
    if width:
        width = size(width)
    height = svgtree.get("height")
    if height:
        height = size(height)

    viewbox = svgtree.get('viewBox')
    if viewbox:
        viewbox = re.sub('[ \n\r\t,]+', ' ', viewbox)
        viewbox = tuple(float(position) for position in viewbox.split())
        width = width or viewbox[2]
        height = height or viewbox[3]

    #print("width {}, height {}".format(width, height))

    # .// *[ @ id = 'g7167']
    xp = svgtree.findall(xpath)
    gc = doElements(xp)
    # gc = doElements(svgtree)

    svg = SVG(gc, width, height)
    return svg

