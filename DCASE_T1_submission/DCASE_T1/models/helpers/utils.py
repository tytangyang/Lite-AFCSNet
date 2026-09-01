"""Small model-construction helpers used by the V4 network."""


def make_divisible(value, divisor=8, min_value=None):
    """Round channel counts using the MobileNet-style 10% rule."""
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)
