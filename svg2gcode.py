#!/usr/bin/env python

import argparse
import sys

from svg import doSVG


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='svg2code', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("svgfile", help="input svg file")
    parser.add_argument("gcodefile", help="output gcode")
    parser.add_argument("--id-re", default=".*", help="process only id matching this re")
    parser.add_argument("--g01", default="F1000 S1000", help="append to first G01 path")
    parser.add_argument("--pre", default="G21\nG92 X0 Y0\nM3\n", help="gcode prelude")
    parser.add_argument("--post", default="M5\n", help="gcode prelude")

    args = parser.parse_args()
    print(args)
    g01append = " "+args.g01
    svg = doSVG(args.svgfile, args.id_re)
    pts = svg.gc.applytransform()
    op = "G00"
    with open(args.gcodefile, "w") as h:
        h.write(args.pre)
        for p in pts:
            if isinstance(p, str):
                if p == "[new-path]":
                    op = "G00"
                    first01 = True
                else:
                    h.write("; {}\n".format(p))
            if isinstance(p, tuple):
                h.write("{} X{} Y{}".format(op, p[0], svg.height-p[1]))
                if first01 and op == "G01":
                    h.write(g01append)
                    first01 = False
                op = "G01"
                h.write("\n")

        h.write(args.post)
