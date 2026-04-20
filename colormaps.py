import numpy as np


def hex2rgb(hex_string):
    """Convert HEX (#RRGGBB) to [r, g, b] floats in [0,1]."""
    if len(hex_string) != 7 or not hex_string.startswith("#"):
        raise ValueError(f"Not a valid HEX color: {hex_string}")
    r = int(hex_string[1:3], 16) / 255.0
    g = int(hex_string[3:5], 16) / 255.0
    b = int(hex_string[5:7], 16) / 255.0
    return np.array([r, g, b])

def rgb2hex(cmap):
    """
    Convert a colormap array (m x 3) to a list of HEX strings (#RRGGBB).
    Accepts floats in [0,1] or 0-255 integer RGB values.
    """
    cmap = np.asarray(cmap, dtype=float)
    if cmap.ndim != 2 or cmap.shape[1] != 3:
        raise ValueError("cmap must be an (N,3) array of RGB values")

    # If values appear in 0-255 range, scale to [0,1]
    if cmap.max() > 1.0 + 1e-8:
        cmap = cmap / 255.0

    cmap = np.clip(cmap, 0.0, 1.0)
    ints = np.rint(cmap * 255).astype(int)
    return [f"#{r:02X}{g:02X}{b:02X}" for r, g, b in ints]



def customcolormap(positions, colors, m=256):
    """
    Create a custom colormap by interpolating between given colors at specified positions.
    Parameters
    ----------
    positions : list or array
        Sorted list of positions (0 → 1), same length as colors.
    colors : list of HEX strings or Nx3 RGB array
        Colors corresponding to each position.
    m : int
        Number of output samples.

    Returns
    -------
    J : ndarray (m × 3)
        The resulting RGB colormap.
    """

    positions = np.array(positions, dtype=float)

    # Validate positions
    if np.min(positions) < 0 or np.max(positions) > 1:
        raise ValueError("Positions must range from 0 to 1.")
    if positions[0] != 0:
        raise ValueError("First position must be 0.")
    if positions[-1] != 1:
        raise ValueError("Last position must be 1.")
    if len(positions) != len(np.unique(positions)):
        raise ValueError("Positions contain duplicates.")

    # Convert colors
    if isinstance(colors, (list, tuple)) and isinstance(colors[0], str):
        rgb_colors = np.array([hex2rgb(c) for c in colors])
    else:
        rgb_colors = np.array(colors, dtype=float)
        if rgb_colors.max() > 1:
            rgb_colors = rgb_colors / 255.0

    if len(rgb_colors) != len(positions):
        raise ValueError("Number of colors must match number of positions.")

    # Prepare output array
    J = np.zeros((m, 3))
    sample_positions = (positions * (m - 1)).astype(int)

    # Assign colors at anchor positions
    J[sample_positions] = rgb_colors

    # Interpolate between colors
    for i in range(len(sample_positions) - 1):
        start, end = sample_positions[i], sample_positions[i + 1]
        if end > start + 1:
            for ch in range(3):
                J[start:end+1, ch] = np.linspace(
                    rgb_colors[i, ch], rgb_colors[i+1, ch], end - start + 1
                )

    # MATLAB applies flipud at end
    return np.flipud(J)


## Colormap definitions (colorbrewer2.org)
cmap = {}

# Custom maps
cmap["Jwave"] = customcolormap(
    positions=[0, 0.5, 1],
    colors=np.array([[0, 0, 130],
                     [203, 203, 203],
                     [134, 0, 0]]) / 255.0
)

cmap["Jlightwave"] = customcolormap(
    positions=[0, 0.5, 1],
    colors=np.array([[0, 0, 1],
                     [1, 1, 1],
                     [1, 0, 0]])
)

cmap["Jcividis"] = customcolormap(
    positions=[0, 0.5, 1],
    colors=np.array([[0, 33, 80],
                     [116, 116, 116],
                     [252, 230, 71]]) / 255.0
)

cmap["Jaurora"] = customcolormap(
    positions=[0, 0.617, 0.77, 1],
    colors=np.array([[255, 255, 255],
                     [29, 134, 133],
                     [115, 0, 231],
                     [0, 0, 135]]) / 255.0
)

cmap["Jmagma"] = customcolormap(
    positions=[0, 0.25, 0.5, 1],
    colors=np.array([[4,   0,  15],
                     [118, 0, 130],
                     [252, 53, 90],
                     [245, 255, 186]]) / 255.0
)

# Inverted maps (flipud equivalent)
cmap["Jwavei"]       = np.flipud(cmap["Jwave"])
cmap["Jlightwavei"]  = np.flipud(cmap["Jlightwave"])
cmap["Jcividisi"]    = np.flipud(cmap["Jcividis"])
cmap["Jaurorai"]     = np.flipud(cmap["Jaurora"])
cmap["Jmagmai"]      = np.flipud(cmap["Jmagma"])

# Colorbrewer sequential
cmap["cbseq6"] = np.flipud(
    np.array([[240, 249, 232],
              [204, 235, 197],
              [168, 221, 181],
              [123, 204, 196],
              [67, 162, 202],
              [8, 104, 172]]) / 255.0
)

# Colorbrewer diverging
cmap["cbdiv9"] = np.array([
    [215,  48,  39],
    [244, 109,  67],
    [253, 174,  97],
    [254, 224, 144],
    [255, 255, 191],
    [224, 243, 248],
    [171, 217, 233],
    [116, 173, 209],
    [69, 117, 180]
]) / 255.0

cmap["cbdiv10"] = np.array([
    [158,  1,  66],
    [213, 62,  79],
    [244, 109,  67],
    [253, 174,  97],
    [254, 224, 139],
    [230, 245, 152],
    [171, 221, 164],
    [102, 194, 165],
    [50,  136, 189],
    [94,   79, 162]
]) / 255.0

# Colorbrewer qualitative
cmap["cbqual4"] = np.array([
    [166, 206, 227],
    [31, 120, 180],
    [178, 223, 138],
    [51, 160, 44]
]) / 255.0

cmap["cbqual9"] = np.array([
    [228,  26,  28],
    [55,  126, 184],
    [77,  175,  74],
    [152,  78, 163],
    [255, 127,   0],
    [255, 255,  51],
    [166,  86,  40],
    [247, 129, 191],
    [153, 153, 153]
]) / 255.0

cmap["cbqual6"] = np.array([
    [166,206,227],
    [31,120,180],
    [178,223,138],
    [51,160,44],
    [251,154,153],    
    [227,26,28]
]) / 255.0

cmap["cbqual7"] = np.array([
    [166,206,227],
    [31,120,180],
    [178,223,138],
    [51,160,44],
    [251,154,153],    
    [227,26,28],
    [253,191,111]
]) / 255.0

cmap["cbqual12"] = np.array([
    [166,206,227],
    [31,120,180],
    [178,223,138],
    [51,160,44],
    [251,154,153],    
    [227,26,28],
    [253,191,111],
    [255,127,0],
    [202,178,214],
    [106,61,154],
    [255,255,153],
    [177,89,40]
]) / 255.0