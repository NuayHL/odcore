import datetime
import logging
import sys

def mylogger(name: str, rank=0, is_initialized=False):
    logger = logging.getLogger(name)
    logger.setLevel("INFO")
    logger.propagate = False

    logfile = "trainingLog/"+name+".txt"

    def _utc8_aera(timestamp):
        now = datetime.datetime.utcfromtimestamp(timestamp) + datetime.timedelta(hours=8)
        return now.timetuple()

    formatter = logging.Formatter('[%(asctime)s]-[%(name)s:%(levelname)s]:%(message)s')
    formatter.converter = _utc8_aera

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # if the rank is 0, then also print the training process on the console
    # if is not in multi GPU training, print on the console
    if not is_initialized or rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

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
