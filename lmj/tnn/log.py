'''Snazzy logging utils by Bryan Silverthorn
(from http://github.com/bsilvert/utcondor).'''

import sys
import curses
import logging


class TTY_Formatter(logging.Formatter):
    '''A log formatter for console output.'''

    _DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    _COLORS = dict(
        TIME='\x1b[00m',
        PROC='\x1b[31m',
        NAME='\x1b[36m',
        LINE='\x1b[32m',
        END='\x1b[00m',
        )

    def __init__(self, stream=None):
        '''Construct this formatter.

        Provides colored output if the stream parameter is specified and is an acceptable TTY.
        We print hardwired escape sequences, which will probably break in some circumstances;
        for this unfortunate shortcoming, we apologize.
        '''
        colors = {k: '' for k in TTY_Formatter._COLORS}
        if stream and hasattr(stream, 'isatty') and stream.isatty():
            curses.setupterm()
            if curses.tigetnum('colors') > 2:
                colors = TTY_Formatter._COLORS
        format = ('%%(levelname).1s '
                  '%(TIME)s%%(asctime)s%(END)s '
                  '%(PROC)s%%(processName)s%(END)s '
                  '%(NAME)s%%(name)s%(END)s:'
                  '%(LINE)s%%(lineno)d%(END)s '
                  '%%(message)s' % colors)
        logging.Formatter.__init__(self, format, TTY_Formatter._DATE_FORMAT)


def get_logger(name=None, level=None, default_level=logging.INFO):
    '''Get or create a logger.'''
    logger = logging.getLogger(name) if name else logging.root

    # set the default level, if the logger is new
    try:
        clean = logger.is_squeaky_clean
    except AttributeError:
        pass
    else:
        if clean and default_level is not None:
            logger.setLevel(logging._levelNames.get(default_level, default_level))

    # unconditionally set the logger level, if requested
    if level is not None:
        logger.setLevel(logging._levelNames.get(level, level))
        logger.is_squeaky_clean = False

    return logger


def enable_default_logging():
    '''Set up logging in the typical way.'''
    get_logger(level='NOTSET')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(TTY_Formatter(sys.stdout))
    logging.root.addHandler(handler)
    return handler
