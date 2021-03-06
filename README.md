# svg2gcode
A python3 program that converts paths of svg files into gcode

svg2gcode.py converts all svg path element in gcode.

Output gcode can be re-arranged ( e.g. resized, offsetted to 0,0) using [gcode-doctor](https://github.com/fzellini/gcode-doctor)

```
python3 svg2gcode.py -h
usage: svg2gcode.py [-h] [--xpath XPATH] [--g01 G01] [--pre PRE] [--post POST]
                    svgfile gcodefile

svg2code

positional arguments:
  svgfile        input svg file
  gcodefile      output gcode

optional arguments:
  -h, --help     show this help message and exit
  --xpath XPATH  process this xpath (e.g. ".//*[@id='g7167']") (default: ./)
  --g01 G01      append to first G01 path (default: F1000 S1000)
  --pre PRE      gcode prelude (default: G21 G92 X0 Y0 M3 )
  --post POST    gcode prelude (default: M5 )
```
