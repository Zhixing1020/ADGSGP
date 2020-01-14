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

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap import semantic as sm

import decimal
decimal.getcontext().prec = 100

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1

def protectedMul(left, right):
    try:
        return left*right
    except OverflowError:
        return 1e7

pset = gp.PrimitiveSet("MAIN", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
#pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedMul, 2, name="mul")
pset.addPrimitive(protectedDiv, 2, name="div")
#pset.addPrimitive(operator.neg, 1)
#pset.addPrimitive(math.cos, 1)
#pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0='x')
pset.renameArguments(ARG1='y')


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, sem_vec=numpy.zeros(180))

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=3, max_=5)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def tarFun(points):
    tar_sem = numpy.array(list(x[0]*x[1]+math.sin((x[0]-1)*(x[1]-1)) for x in points))
    return tar_sem

def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    #sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    semanVe = numpy.array(list(func(x[0], x[1]) for x in points))
    sqerrors = tuple((decimal.Decimal(func(x[0], x[1])) - decimal.Decimal(x[0]*x[1]+math.sin((x[0]-1)*(x[1]-1)))) ** 2 for x in points)
    res = (math.fsum(sqerrors) / len(points), semanVe)
    return (math.fsum(sqerrors) / len(points),semanVe),

#toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,10)])
indata=[]
for i in range(0,180):
    x=random.randrange(-10,10)/10.
    y=random.randrange(-10,10)/10.
    indata.append((x,y))
toolbox.register("evaluate", evalSymbReg, points=indata)
toolbox.register("select", tools.selTournament, tournsize=3)
#toolbox.register("mate", gp.cxOnePoint)
#toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
#toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
#toolbox.register("SCX", gp.cxSemanticTest, pset=pset)
#toolbox.register("SMT", gp.mutSemanticTest, pset=pset)
toolbox.register("SCR", sm.semConRep, toolbox=toolbox, pset=pset, points=indata)
toolbox.register("ADS", sm.angleDrivenSel, toolbox=toolbox)
toolbox.register("PC", sm.perpendicularCX)
toolbox.register("RSM", sm.randSegMut)

#toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
#toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("SCR", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
#toolbox.decorate("SCX", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
#toolbox.decorate("SMT", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

def main():
    random.seed(318)

    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(1)

    #test
    #lib = sm.library(toolbox, pset=pset, points=indata)
    #testa=toolbox.expr()
    #lib.lib_maintain(pop)
    #ex=creator.Individual(toolbox.expr())
    #print(ex)
    #ex3=sm.semConRep(ex,17,AL=[],tarSem=tarFun(indata),toolbox=toolbox,pset = pset,points = indata, library=lib)
    #print(ex3)
    #parent, children = sm.decode(ex)
    #nst = [pset.mapping['add'], 20, pset.mapping['mul'], 10, ex]
    #ex2 = creator.Individual(nst)
    #print(ex2)
    #print(creator.Individual(ex[ex.searchSubtree(4)]).__str__())
    #code = str(ex)
    #print(gp.compile(code, pset))
    #sub_tree=sm.subTree(toolbox, ex, pset, indata)
    #lib.insert_lib(sub_tree)
    #lib.lib_maintain(pop)
    #tar_sem = tarFun(points=indata)
    #sel_pairs = sm.angleDrivenSel(toolbox,pop=pop,tarSem=tar_sem, np=5,nt=5,ta=90)
    #dsr_sem=[]
    #print(numpy.hstack([dsr_sem,0]))
    #print(ex[3])

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    lib = sm.library(toolbox, pset=pset, points=indata)
    lib.lib_maintain(pop)

    pop, log = algorithms.semanticGP(pop, toolbox, tarSem=tarFun(indata), library=lib, ngen=1000, stats=mstats,
                                   halloffame=hof, verbose=True)
    print(hof.items[0])
    print(hof.items[0].fitness)

    return pop, log, hof

if __name__ == "__main__":
    main()
