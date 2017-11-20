from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time
from logzero import logger


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            logger.debug('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed
