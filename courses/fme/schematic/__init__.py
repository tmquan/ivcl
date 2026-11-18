"""schematic: circuit symbols measured from Razavi's "Fundamentals of Microelectronics",
a tiny schematic layout layer, and renderers (matplotlib, SVG, draw.io).

    from schematic import show, symbol_sheet
    def inverter(c): ...
    show(inverter)            # drawn with class BJT on the left, with class MOSFET on the right
"""
from . import symbols, scene, tex, poly
from .scene import Scene, RED
from .ckt import Ckt, Transistor, BJT, MOSFET, mid

__all__ = ['symbols', 'scene', 'tex', 'poly', 'Scene', 'RED', 'Ckt', 'Transistor', 'BJT', 'MOSFET', 'mid', 'show', 'grid', 'panels', 'build', 'draw', 'names', 'symbol_sheet']


def __getattr__(name):          # matplotlib is imported only when a renderer is used
    if name in ('show', 'grid', 'panels', 'build', 'draw', 'names', 'symbol_sheet'):
        from . import mpl
        return getattr(mpl, name)
    raise AttributeError(name)
