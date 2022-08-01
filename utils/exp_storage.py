import os
import datetime
import logging
import sys

def mylogger(loggerfilename: str, file_path=''):
    logger = logging.getLogger(loggerfilename)
    logger.setLevel("INFO")
    logger.propagate = False

    logfile = os.path.join(file_path, loggerfilename+'.log')

    def _utc8_aera(timestamp):
        now = datetime.datetime.utcfromtimestamp(timestamp) + datetime.timedelta(hours=8)
        return now.timetuple()

    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]:%(message)s')
    formatter.converter = _utc8_aera

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

# for fun
def printlogto(filename, file_path="trainingLog/"):
    def change_std_out(fun):
        def decoratedFun(*args, **kwargs):
            ori_stdout = sys.stdout
            with open(file_path+filename, "a") as f:
                sys.stdout = f
                returned = fun(*args, **kwargs)
                sys.stdout = ori_stdout
            return returned
        return decoratedFun
    return change_std_out
