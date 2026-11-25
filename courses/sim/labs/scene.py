"""scene: the scene of Chapter 0, written down once.

Every script of this chapter builds its scene from here, so they all
show the same cube, on the same ground, from the same camera.
"""
import look

SIZE = 0.3                      # cube edge, metres
START = (0.0, 0.0, 1.0)         # cube centre at the start, metres
VIEW = look.View(eye=(2.2, -2.7, 1.9), target=(0.0, 0.0, 0.5))


def build(centre=START):
    """A 3 m square of ground and a green cube above it."""
    return [look.ground(half=1.5),
            look.box("cube", centre, SIZE, look.GREEN_RGB)]
