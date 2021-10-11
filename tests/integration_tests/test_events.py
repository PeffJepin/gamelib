from collections import defaultdict

import pytest

from src.gamelib import KeyDown, ModifierKeys, Keys
from src.gamelib.events import (
    eventhandler,
    Event,
    post_event,
    register_marked,
)
