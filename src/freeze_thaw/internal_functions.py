from freeze_thaw.constants import constants as c

"""
Internal functions used in src.
"""

def classify_value(x):
    """
    Return class label of x based on boundary and classes.
    For use in .map()
    """
    if x > abs(c.CLASS_BOUNDARY):
        return c.CLASSES[0]
    elif x >= -abs(c.CLASS_BOUNDARY):
        return c.CLASSES[1]
    else:
        return c.CLASSES[2]