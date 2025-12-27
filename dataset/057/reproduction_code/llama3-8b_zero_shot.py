Here is a minimal, self-contained Python script to reproduce the bug:

```
`python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import pathlib
import random
import string
import re
import numpy as np

import tensorflow.data as tf_data
import tensorflow.strings as tf_strings

import keras
from keras import layers
try:
    from keras import ops
except ImportError:
    pass

from keras.layers import TextVectorization
```

