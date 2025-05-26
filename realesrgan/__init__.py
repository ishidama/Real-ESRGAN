# flake8: noqa
from .archs import *
from .data import *
from .models import *
from .utils import *

# バージョン情報（オプション）
try:
    from .version import *
except ImportError:
    pass

# CoreMLのインポート（条件付き）
try:
    from .utils_coreml import COREML_AVAILABLE, RealESRGANerCoreML
except ImportError:
    # CoreMLが利用できない場合はダミークラスを提供
    COREML_AVAILABLE = False
    RealESRGANerCoreML = None
