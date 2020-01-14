"""The :mod:`semantic` module provides the methods and classes to perform
Genetic Programming with Semantic backpropagation with DEAP. It essentially contains the classes to
evaluate the gp tree, maintain the semantic library, and variate the gp tree.
"""

from deap import gp
import math
import random
import numpy
from deap import creator
from deap import base


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

def angle_dis(vec1, vec2):
    if len(vec1) != len(vec2):
        print("the dimension of two vectors must be consistent in angle_dis function")
        return 0
    vec1_sig = vec1.sum()  # math.fsum(vec1)
    vec2_sig = vec2.sum()  # math.fsum(vec2)
    vec_sum = (vec1*vec2).sum()
    norm_vec1 = math.sqrt((vec1*vec1).sum())  # math.fsum((vi**2 for vi in vec1))
    norm_vec2 = math.sqrt((vec2*vec2).sum())  # math.fsum((vi ** 2 for vi in vec2))
    res = vec_sum/(norm_vec1*norm_vec2)
    if res>1:
        res=1
    elif res<-1:
        res=-1
    return math.acos(res)

class Node(object):
    content =[]  #primitive/terminal/ephemeral
    index=[]
    subSem0=[]  #numpy.ndarray  sub semantic
    subSem1=[]

    def __init__(self, con, index, subSem0, subSem1):
        self.content=con
        self.index=index
        self.subSem0=subSem0
        self.subSem1=subSem1

    def Invert(self, con, k, tarSem):
        """
        con: the node content
        k: the index of param 0 / 1
        tarSem: desired semantic
        """
        dsr_sem=[]
        if con.name() == "add":
            if k==0:
                dsr_sem = tarSem - self.subSem1
            else:
                dsr_sem = tarSem - self.subSem0
        elif con.name()=="sub":
            if k==0:
                dsr_sem = tarSem + self.subSem1
            else:
                dsr_sem = self.subSem0 - tarSem
        if con.name() == "mul":
            if k==0:
                for i in range(len(tarSem)):
                    if self.subSem1[i]!=0:
                        numpy.hstack([dsr_sem, tarSem[i]/self.subSem1[i]])
                    if self.subSem1[i]==0 and tarSem[i]==0:
                        numpy.hstack([dsr_sem, tarSem[i]])
                    if self.subSem1[i]==0 and tarSem[i]!=0:
                        numpy.hstack([dsr_sem, 0])
            else:
                for i in range(len(tarSem)):
                    if self.subSem0[i]!=0:
                        numpy.hstack([dsr_sem, tarSem[i]/self.subSem0[i]])
                    if self.subSem0[i]==0 and tarSem[i]==0:
                        numpy.hstack([dsr_sem, tarSem[i]])
                    if self.subSem0[i]==0 and tarSem[i]!=0:
                        numpy.hstack([dsr_sem, 0])
        if con.name()=="div":
            if k==0:
                for i in range(len(tarSem)):
                    if math.isfinite(self.subSem1[i]):
                        numpy.hstack([dsr_sem, tarSem[i]*self.subSem1[i]])
                    if math.isinf(self.subSem1[i]) and tarSem[i]==0:
                        numpy.hstack([dsr_sem, tarSem[i]])
                    if math.isinf(self.subSem1[i]) and tarSem[i]!=0:
                        numpy.hstack([dsr_sem, 0])
            else:
                for i in range(len(tarSem)):
                    if self.subSem0[i]!=0:
                        numpy.hstack([dsr_sem, self.subSem0[i]/tarSem[i]])
                    if self.subSem0[i]==0 and tarSem[i]==0:
                        numpy.hstack([dsr_sem, tarSem[i]])
                    if self.subSem0[i]==0 and tarSem[i]!=0:
                        numpy.hstack([dsr_sem, 0])

        return dsr_sem

def Invert(con, k, tarSem, subSem0=numpy.zeros(1), subSem1=numpy.zeros(1)):
    """
    con: the node content
    k: the index of param 0 / 1
    tarSem: desired semantic
    """
    dsr_sem=numpy.zeros(tarSem.size)
    if con.name == "add":
        if k==0:
            dsr_sem = tarSem - subSem1
        else:
            dsr_sem = tarSem - subSem0
    elif con.name=="sub":
        if k==0:
            dsr_sem = tarSem + subSem1
        else:
            dsr_sem = subSem0 - tarSem
    if con.name == "mul":
        if k==0:
            for i in range(len(tarSem)):
                if subSem1[i]!=0:
                    dsr_sem[i] = tarSem[i]/subSem1[i]
                if subSem1[i]==0 and tarSem[i]==0:
                    dsr_sem[i] = tarSem[i]
                if subSem1[i]==0 and tarSem[i]!=0:
                    dsr_sem[i] = 0
        else:
            for i in range(len(tarSem)):
                if subSem0[i]!=0:
                    dsr_sem[i] = tarSem[i]/subSem0[i]
                if subSem0[i]==0 and tarSem[i]==0:
                    dsr_sem[i] = tarSem[i]
                if subSem0[i]==0 and tarSem[i]!=0:
                    dsr_sem[i] = 0
    if con.name =="div":
        if k==0:
            for i in range(len(tarSem)):
                if math.isfinite(subSem1[i]):
                    dsr_sem[i] = tarSem[i]*subSem1[i]
                if math.isinf(subSem1[i]) and tarSem[i]==0:
                    dsr_sem[i] = tarSem[i]
                if math.isinf(subSem1[i]) and tarSem[i]!=0:
                    dsr_sem[i] = 0
        else:
            for i in range(len(tarSem)):
                if subSem0[i]!=0 and tarSem[i]!=0:
                    dsr_sem[i] = subSem0[i]/tarSem[i]
                elif subSem0[i]==0 and tarSem[i]==0:
                    dsr_sem[i] = tarSem[i]
                elif subSem0[i]==0 and tarSem[i]!=0:
                    dsr_sem[i] = 0

    return dsr_sem

def decode(ind):
    """
    ind: type individual
    return:
    parent list (each item: Node type (primitive/terminal, index))
    children list (Node type)
    """
    parent= [] #1 dimension list
    children= []  #2 dimension list
    pi=-1   #pi: the processing index;   usi: the last unsatisfied index
    parent.append(pi)
    pi = usi = pi + 1
    if len(ind) == 1:
        return parent, children

    for i in range(1, len(ind)):

        #update the parent
        parent.append(pi)    #record the parent of current node
        #pi = pi + 1          #update the parent

        #update the children
        if pi + 1 > len(children):  # if the parent node is new, append new item to the tail of children
            children.append([i])

        # if the parent node has been met, append the item to the existing item of children
        else:
            children[pi].append(i)

        #update the usi
        #if pi is satisfied
        if ind[pi].arity == len(children[pi]):
            #find the last unsatisfied parent
            while ind[usi].arity <= len(children[usi]):
                if usi >= 1:
                    usi = usi - 1
                else:
                    break
        #if pi hasn't been satisfied
        else:
            usi = pi

        #update the pi
        #if i is the primitive, pi follows i
        if ind[i].arity > 0:
            pi = usi = i

        #else if i is the terminal / ephemeral && pi is not satisfied, pi keeps still
        #else if i is the terminal / ephemeral && pi is satisfied, pi follows usi
        elif ind[i].arity==0 and ind[pi].arity == len(children[pi]):
            pi = usi

        #if i is the terminal / ephemeral, insert the placeholder into children
        if ind[i].arity == 0:
            children.append([-1,-1])


    return parent, children



class subTree(object):
    expr=[]
    sem_vec=[]
    angle_dis=-1

    def __init__(self, toolbox, expr_st, points):
        """expr_st: the sub tree individual (Individual object)"""
        self.expr = expr_st
        func = toolbox.compile(expr=expr_st)
        # Evaluate the mean squared error between the expression
        # and the real function : x**4 + x**3 + x**2 + x
        # sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
        sqerrors = numpy.array(list(func(x[0], x[1]) for x in points))
        self.sem_vec=sqerrors
        #print(self.sem_vec)

def angleDrivenSel(toolbox, pop, tarSem, np, nt, ta):
    """pop: population
        tarSem: target semantics
        np: the number of pairs
        nt: maximum number of trials
        ta: the thresold of angle distance
        return: a list of selected pairs
    """
    output_list=[]
    for i in range(0, np):
        flag = False
        p1 = toolbox.select(pop, k=1)[0]
        cp2_use = toolbox.select(pop, k=1)[0]
        maxangle = -500
        for j in range(0, nt):
            cp2 = toolbox.select(pop, k=1)[0]
            relSV1 = tarSem - p1.sem_vec
            relSV2 = tarSem - cp2.sem_vec
            gamma = angle_dis(relSV1, relSV2)
            if gamma > ta:
                p2=cp2
                flag=True
                break
            else:
                if gamma > maxangle:
                    cp2_use=cp2
                    maxangle=gamma
                #else:
                    #print(gamma, " ", cp2.sem_vec)

        if flag==False:
            p2=cp2_use

        output_list.append((p1,p2))

    return output_list

def perpendicularCX(parents, tarSem):
    p1=parents[0]
    p2=parents[1]
    relSV1 = tarSem - p1.sem_vec # list(tarSem[i] - p1.sem_vec[i] for i in range(len(tarSem)))
    relSV2 = tarSem - p2.sem_vec # list(tarSem[i] - p2.sem_vec[i] for i in range(len(tarSem)))
    relatSV1 = p2.sem_vec - p1.sem_vec  # list(p2.sem_vec[i] - p1.sem_vec[i] for i in range(len(tarSem)))
    relatSV2 = p1.sem_vec - p2.sem_vec  # list(p1.sem_vec[i] - p2.sem_vec[i] for i in range(len(tarSem)))
    alpha = angle_dis(relSV1, relatSV1)
    beta = angle_dis(relSV2, relatSV2)

    relaNorm = math.sqrt((relatSV1**2).sum())
    if alpha <= 90 and beta < 90:
        roNorm = math.sqrt(((p1.sem_vec - tarSem)**2).sum())*math.cos(alpha)
        ov = p1.sem_vec + (roNorm /relaNorm) * relatSV2
    elif alpha > 90:
        roNorm = math.sqrt(((p1.sem_vec - tarSem) ** 2).sum()) * math.cos(180-alpha)
        ov = p1.sem_vec - (roNorm / relaNorm) * relatSV2
    elif beta >= 90:
        roNorm =  math.sqrt(((p2.sem_vec - tarSem) ** 2).sum()) * math.cos(180-beta)
        ov = p2.sem_vec + (roNorm /relaNorm) * relatSV2
    else:
        ov = randSegMut(p1, tarSem)

    return ov

def randSegMut(parent, tarSem):
    relaSV = tarSem - parent.sem_vec
    k = random.random()
    return parent.sem_vec + k*relaSV

def semConRep(par, tarSem, toolbox, pset, points, library):
    """ par: the parent individual
    MD: the maximum depth
    AL: the angle distance list
    tarSem: the final desired semantic
    """
    curDep = par.__len__()
    spt = random.randint(0, curDep - 1)  #spliting point
    while par[spt].arity==0:
        spt = random.randint(0, curDep - 1)  # spliting point

    node = par[spt]

    #====replace the spliting point====
    prims = [p for p in pset.primitives[node.ret] if p.args == node.args]
    par[spt] = random.choice(prims)
    #=====excute the inverse operation====
    #decode the individual
    parent_arr, children_arr=decode(par)

    #identify the path from root to R
    path=[spt]
    travel=spt
    while parent_arr[travel] != -1:
        path.append(parent_arr[travel])
        travel = parent_arr[travel]
    path.reverse()
    #based on the children_arr, construct the other subtree and get its semantic
    dsr_sem=tarSem
    if len(path)>1:
        for j in path[1:len(path)-1]:
            #R in left sub tree
            if j == children_arr[parent_arr[j]][0]:
                #calculate the semantic of right sub tree
                fix_st=children_arr[parent_arr[j]][1]
                sub_tree=subTree(toolbox,creator.Individual(par[par.searchSubtree(fix_st)]),points)
                dsr_sem = Invert(par[j], 0, dsr_sem, subSem1=sub_tree.sem_vec)
            else: #R in right sub tree
                #calculate the semantic of left sub tree
                fix_st=children_arr[parent_arr[j]][0]
                sub_tree = subTree(toolbox, creator.Individual(par[par.searchSubtree(fix_st)]), points)
                dsr_sem = Invert(par[j], 1, dsr_sem, subSem0=sub_tree.sem_vec)

    #randomly select a child serving as T, if node R is a primitive.
    #determine the sub-semantic
    if node.arity>0:
        prefix = random.randint(0, node.arity - 1)
        #Snode = children_arr[parent_arr[path[-1]]][prefix]
        Snode = children_arr[spt][prefix]
        if prefix == 1:   #left subtree is T
            #fix_st = children_arr[parent_arr[path[-1]]][1]
            fix_st = children_arr[spt][1]
            sub_tree = subTree(toolbox, creator.Individual(par[par.searchSubtree(fix_st)]), points)
            dsr_sem = Invert(node, 0, dsr_sem, subSem1=sub_tree.sem_vec)
        else:
            #fix_st = children_arr[parent_arr[path[-1]]][0]
            fix_st = children_arr[spt][0]
            sub_tree = subTree(toolbox, creator.Individual(par[par.searchSubtree(fix_st)]), points)
            dsr_sem = Invert(node, 1, dsr_sem, subSem0=sub_tree.sem_vec)
    else:
        return par   #cause the node R (a termnal / ephemeral) has been modified by a random operation at the beginning

    #====construct the angle list from the library====
    min_angle=500
    a=b=float()
    min_st = library.expr_pool[0].expr
    for st in library.expr_pool:
        #compute the angle
        st.angle_dis = angle_dis(st.sem_vec, dsr_sem)
        if st.angle_dis < min_angle:
            min_angle = st.angle_dis
            min_st = st.expr
            #obtain the coefficient b
            if ((st.sem_vec - st.sem_vec.sum()/st.sem_vec.size)**2).sum() != 0:
                b = ((dsr_sem - dsr_sem.sum()/dsr_sem.size)*(st.sem_vec - st.sem_vec.sum()/st.sem_vec.size)).sum()/((st.sem_vec - st.sem_vec.sum()/st.sem_vec.size)**2).sum()
            else:
                b = ((dsr_sem - dsr_sem.sum() / dsr_sem.size) * (st.sem_vec - st.sem_vec.sum() / st.sem_vec.size)).sum()
            #obtain the coefficient a
            a = dsr_sem.sum()/dsr_sem.size - b * st.sem_vec.sum()/st.sem_vec.size

    #====crossover into the subtree====
    #use a, b construct the new sub tree

    nst =[pset.mapping['add'], gp.Constant(a), pset.mapping['mul'], gp.Constant(b)]
    nst.extend(min_st[:])
    new_subtree = creator.Individual(nst)

    CT_slice = par.searchSubtree(Snode)
    #par[CT_slice] = creator.Individual(min_st)
    par[CT_slice] = new_subtree

    return par,


class library(object):

    expr_pool=[]  #list of subTree type items
    toolbox=[]
    pset=[]
    points=[]

    def __init__(self, toolbox, pset, points):
        self.toolbox=toolbox
        self.pset=pset
        self.points=points
        print("init library")


    def similarity(self, sv1, sv2):
        """compare the similarity between two semantic vector"""
        return ((sv1-sv2)**2).sum()/sv1.size

    def insert_lib(self, subt):
        """subt is the subTree type"""
        self.expr_pool.append(subt)

    def lib_clear(self):
        self.expr_pool.clear()

    def lib_maintain(self, pop):
        """pop: the population"""
        self.lib_clear()
        for chro in pop:
            #extract any sub tree from chro
            #check semantically unique
            #insert into the library
            for inde in range(1, chro.__len__()):
                sub_expr=creator.Individual(chro[chro.searchSubtree(inde)])
                sub_tree=subTree(self.toolbox, sub_expr, self.points)
                if len(self.expr_pool)==0:
                    should_insert = True
                else:
                    should_insert=all((self.similarity(sub_tree.sem_vec, st.sem_vec) for st in self.expr_pool))
                if should_insert:
                    self.insert_lib(sub_tree)