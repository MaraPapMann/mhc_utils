'''
@Author: MHC
@Desc: A logger
'''
import sys
sys.path.append('../..')
import utils.operator as opt


class Logger():
    def __init__(self, dir_log:str):
        if not opt.os.pth_exist(dir_log):
            opt.os.mkdir(dir_log)
        self.pth_log = opt.os.join(dir_log, 'log.txt')
        self.f = open(self.pth_log, 'w')
        return
    
    def log(self, msg:str, newline:bool=True, verbose:bool=True):
        self.f.write(msg)
        if newline:
            self.f.write('\n')
        if verbose:
            print(msg)
        return
    
    def end(self):
        self.f.close()
        return