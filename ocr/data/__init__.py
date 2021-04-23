from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
from ocr.data.imaug import transform, create_operators


