#!/usr/bin/env python3
"""兼容入口：请优先使用 ``yolo11-convert-pt`` 或 ``python -m model.convert_ultralytics_weights``。"""

from model.convert_ultralytics_weights import main

if __name__ == "__main__":
    main()
