#    This file is for testing
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

import operator
import math
import random
import time
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap import semantic as sm
from deap import StatisticFile

import decimal
decimal.getcontext().prec = 100

# Define new functions
def protectedDiv(left, right):
    if right == 0:
        return left
    res = left / right
    if res > 1e7:
        return 1e7
    if res < -1e7:
        return -1e7
    return res

def protectedMul(left, right):
    try:
        return left*right
    except OverflowError:
        return 1e7

def protectedLog(arg):
    if abs(arg) < 1e-5:
        arg = 1e-5
    return math.log(abs(arg))

def protectedExp(arg):
    if arg > 10:
        arg = 10
    return math.exp(arg)


pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
#pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedMul, 2, name="mul")
pset.addPrimitive(protectedDiv, 2, name="div")
#pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1, name="cos")
pset.addPrimitive(math.sin, 1, name="sin")
pset.addPrimitive(protectedExp, 1, name= "exp")
pset.addPrimitive(protectedLog, 1, name="ln")
#pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
#pset.renameArguments(ARG0='x')
#pset.renameArguments(ARG1='y')


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, sem_vec=numpy.zeros(180))

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points, outputs):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    #sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    #semanVe = numpy.array(list(func(x[0], x[1]) for x in points))
    semanVe = numpy.array(list(func(x[0]) for x in points))
    sqerrors = []
    for x, y in zip(points, outputs):
        #sqerrors.append((decimal.Decimal(func(x[0], x[1])) - decimal.Decimal(y)) ** 2)
        sqerrors.append((decimal.Decimal(func(x[0])) - decimal.Decimal(y)) ** 2)

    return (math.sqrt(math.fsum(sqerrors) / len(points)),semanVe),

toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("selectBest", tools.selBest)
toolbox.register("ADS", sm.angleDrivenSel, toolbox=toolbox)
toolbox.register("PC", sm.perpendicularCX)
toolbox.register("RSM", sm.randSegMut)

def loadData(flieName):
  inFile = open(flieName, 'r')
  X = []
  y = []
  datasize = int(inFile.readline().split('\t')[0])
  for i in range(0,datasize):
      line=inFile.readline()
      trainingSet = line.split('\t')
      x=[]
      for xi in trainingSet[:-1]:
          x.append(float(xi))
      X.append(tuple(x))
      y.append(float(trainingSet[-1]))
  return X, y

def main():

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    avg_fit = 0
    suc_rate = 0
    avg_size = 0

    SF = StatisticFile.StatisticFile()

    for run in range(0,4):
        SF.problemInit(30)
        for ei in range(0, 30):
            random.seed(318 + ei)

            SF.independentRunInit(run, ei)
            suc = 0
            eva_times = 0
            train_err = 1e6
            test_err = 1e6
            prog_size = 0
            traintime = 0

            filename = "D:\\exp_data\\SLGP_for_SR\\F%d_%d_training_data.txt" % (run, ei % 10)
            indata, outdata = loadData(filename)
            #remember to update the __init__ function of class subTree

            toolbox.register("evaluate", evalSymbReg, points=indata, outputs=outdata)
            toolbox.register("SCR", sm.semConRep, toolbox=toolbox, pset=pset, points=indata)
            toolbox.decorate("SCR", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

            time_start = time.time()

            pop = toolbox.population(n=100)
            hof = tools.HallOfFame(1)
            lib = sm.library(toolbox, pset=pset, points=indata)

            pop, log, eva_times = algorithms.semanticGP(pop, toolbox, tarSem=outdata, library=lib, output_file=SF, ngen=100000,
                                             stats=mstats, halloffame=hof, verbose=True)

            time_end = time.time()

            if hof.items[0].fitness.values[0] < 1e-4:
                suc = 1
            train_err = hof.items[0].fitness.values[0]
            prog_size = hof.items[0].__len__()
            traintime = time_end - time_start

            #testing
            filename = "D:\\exp_data\\SLGP_for_SR\\F%d_%d_testing_data.txt" % (run, ei % 10)
            indata, outdata = loadData(filename)
            test_err = evalSymbReg(hof.items[0], indata, outdata)[0][0]

            print(hof.items[0])
            print(hof.items[0].fitness)

            SF.independentRunRecord(run,ei,suc,eva_times,train_err,test_err,prog_size,hof.items[0].__str__(),traintime)
        SF.problemOverallRecord(run)


    return pop, log, hof

if __name__ == "__main__":
    main()
