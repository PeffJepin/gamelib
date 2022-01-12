import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class Model:
    vertices: np.ndarray
    normals: Optional[np.ndarray]
    triangles: np.ndarray
