# coding: utf-8

import platform

GPU = True

if platform.system() == "Darwin":
    GPU = False


