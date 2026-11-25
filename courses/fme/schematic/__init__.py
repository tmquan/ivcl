"""schematic: circuit symbols as printed in the standard textbooks,
a tiny schematic layout layer, and renderers (Plotly canvas, matplotlib, SVG, draw.io).

    from schematic import show, symbol_sheet
    def inverter(c): ...
    show(inverter)            # drawn with class BJT on the left, with class MOSFET on the right;
                              # in a notebook on an interactive Plotly canvas, in a script with matplotlib
"""
from . import symbols, scene, tex, poly
from .scene import Scene, RED
from .ckt import Ckt, Transistor, BJT, MOSFET, mid

__all__ = ['symbols', 'scene', 'tex', 'poly', 'Scene', 'RED', 'Ckt', 'Transistor', 'BJT', 'MOSFET', 'mid', 'show', 'grid', 'panels', 'build', 'draw', 'names', 'symbol_sheet', 'gallery']


def __getattr__(name):          # matplotlib and Plotly are imported only when a renderer is used
    if name in ('show', 'symbol_sheet', 'gallery'):                 # the interactive canvas; mpl.show draws the same figure as a picture
        from . import canvas
        return getattr(canvas, name)
    if name in ('grid', 'panels', 'build', 'draw', 'names'):
        from . import mpl
        return getattr(mpl, name)
    raise AttributeError(name)
