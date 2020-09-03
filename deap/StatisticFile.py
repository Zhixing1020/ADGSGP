from __future__ import division
import numpy as np

class StatisticFile(object):
    suc = []
    train_error = []
    test_error = []
    eva_times = []
    prog_size = []
    trainTime = []

    conv_internal = 50000
    last_internal = 0

    overallAdd = "..\\overall.txt"
    fitness_convAdd = "..\\fitness_conv.txt"
    fitness_timesAdd = "..\\fitness_time.txt"
    train_errorAdd = "..\\train_error.txt"
    test_errorAdd = "..\\test_error.txt"
    prog_sizeAdd = "..\\prog_size.txt"
    example_progAdd = "..\\example_prog.txt"
    trainTimeAdd = "..\\trainTime.txt"

    def __init__(self):
        self.initFile()

    def initFile(self):
        doc = open(self.overallAdd, 'w')
        doc.close()
        doc = open(self.fitness_convAdd, 'w')
        doc.close()
        doc = open(self.fitness_timesAdd, 'w')
        doc.close()
        doc = open(self.train_errorAdd, 'w')
        doc.close()
        doc = open(self.test_errorAdd, 'w')
        doc.close()
        doc = open(self.prog_sizeAdd, 'w')
        doc.close()
        doc = open(self.example_progAdd, 'w')
        doc.close()
        doc = open(self.trainTimeAdd, 'w')
        doc.close()

    def problemInit(self, iterative):
        self.suc = np.zeros(iterative)
        self.train_error = np.zeros(iterative)
        self.test_error = np.zeros(iterative)
        self.eva_times = np.zeros(iterative)
        self.prog_size = np.zeros(iterative)
        self.trainTime = np.zeros(iterative)

    def independentRunInit(self, run, job):
        self.last_internal = 0

        doc = open(self.fitness_convAdd, 'a')
        print("\n%d\t%d" % (run, job), file=doc)
        doc.close()
        doc = open(self.fitness_timesAdd, 'a')
        print("%d\t%d\t" % (run, job), file=doc, end='')
        doc.close()
        doc = open(self.train_errorAdd, 'a')
        print("%d\t%d\t" % (run, job), file=doc, end='')
        doc.close()
        doc = open(self.test_errorAdd, 'a')
        print("%d\t%d\t" % (run, job), file=doc, end='')
        doc.close()
        doc = open(self.prog_sizeAdd, 'a')
        print("%d\t%d\t" % (run, job), file=doc, end='')
        doc.close()
        doc = open(self.example_progAdd, 'a')
        print("%d\t%d" % (run, job), file=doc)
        doc.close()
        doc = open(self.trainTimeAdd, 'a')
        print("%d\t%d\t" % (run, job), file=doc, end='')
        doc.close()

    def procedureRecord(self, eva, fitness):
        doc = open(self.fitness_convAdd, 'a')
        recordeval = eva // self.conv_internal
        if self.last_internal == 0:
            print("%d\t%f"%(self.last_internal, fitness), file=doc)
        for i in range(1+self.last_internal, recordeval + 1):
            print("%d\t%f" % (i*self.conv_internal, fitness), file=doc)
        self.last_internal = recordeval
        doc.close()

    def independentRunRecord(self, run, job, sucFlag, f_times, train_err, test_err, prog_siz, example_prog, train_time):
        self.suc[job] = sucFlag
        self.eva_times[job] = f_times
        self.train_error[job] = train_err
        self.test_error[job] = test_err
        self.prog_size[job] = prog_siz
        self.trainTime[job] = train_time

        print("%d\t%d\tsuc:\t%f" % (run, job, self.suc.sum() / (job + 1)), end='')
        print("\teva_times:\t%f" % (self.eva_times.sum() / (job+1)), end='')
        print("\ttrain_error:\t%f" % (self.train_error.sum() / (job+1)), end='')
        print("\ttest_error:\t%f" % (self.test_error.sum() / (job + 1)), end='')
        print("\tprog_size:\t%f" % (self.prog_size.sum() / (job + 1)), end='')
        print("\ttrain_time:\t%f" % (self.trainTime.sum() / (job + 1)))

        doc = open(self.fitness_timesAdd, 'a')
        print("%f" % (f_times), file=doc)
        doc.close()
        self.procedureRecord(f_times,train_err)
        doc = open(self.train_errorAdd, 'a')
        print("%f" % (train_err), file=doc)
        doc.close()
        doc = open(self.test_errorAdd, 'a')
        print("%f" % (test_err), file=doc)
        doc.close()
        doc = open(self.prog_sizeAdd, 'a')
        print("%f" % prog_siz, file=doc)
        doc.close()
        doc = open(self.example_progAdd, 'a')
        print("%s\n" % example_prog, file=doc)
        doc.close()
        doc = open(self.trainTimeAdd, 'a')
        print("%f" % train_time, file=doc)
        doc.close()

    def problemOverallRecord(self, run):
        doc = open(self.overallAdd,'a')
        print("%d\tsuc:\t%f" % (run, self.suc.sum() / self.suc.size), end='', file=doc)
        print("\teva_times:\t%f" % self.eva_times.sum() / self.eva_times.size, end='', file=doc)
        print("\ttrain_error:\t%f" % self.train_error.sum() / self.train_error.size, end='', file=doc)
        print("\ttest_error:\t%f" % self.test_error.sum() / self.test_error.size, end='', file=doc)
        print("\tprog_size:\t%f" % self.prog_size.sum() / self.prog_size.size, end='', file=doc)
        print("\ttrain_time:\t%f" % self.trainTime.sum() / self.trainTime.size, file=doc)
        doc.close()
