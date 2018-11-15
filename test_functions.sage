# Make sure current directory is in path.  
# That's not true while doctesting (sage -t).
if '' not in sys.path:
    sys.path = [''] + sys.path

from igp import *

#import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import csv
import time
import os
import glob
from sage.misc.sage_timeit import sage_timeit

random.seed(500)

default_function_name_list=['chen_4_slope','drlm_backward_3_slope','gj_forward_3_slope','kzh_3_slope_param_extreme_1','kzh_3_slope_param_extreme_2','kzh_4_slope_param_extreme_1','kzh_7_slope_1','kzh_7_slope_2','kzh_7_slope_3','kzh_7_slope_4','kzh_10_slope_1','kzh_28_slope_1','kzh_28_slope_2','gj_2_slope_repeat','bcdsp_arbitrary_slope']
default_two_slope_fill_in_epsilon_list=[1/(i*10) for i in range(1,11)]
default_perturbation_epsilon_list=[i/100 for i in range(3)]
default_max_number_of_bkpts=[0,10,20,40,100,400,1000,10000,100000]

def merge_csv_files(filename,header):
    """
    Merge all .csv files with the same header to one .csv file.
    """
    allFiles = glob.glob(os.path.join("*.csv"))
    np_array_list = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=int(0))
        np_array_list.append(df.as_matrix())
    comb_np_array = np.vstack(np_array_list)
    big_frame = pd.DataFrame(comb_np_array)
    big_frame.columns = header
    big_frame.to_csv(filename)

def generate_mip_of_delta_pi_min_pulp_dlog(fn):
    """
    Generate the Disaggregated Logarithmic mip formulation of computing the minimum of delta pi. Using pulp with COIN_CMD.
    """
    bkpts=fn.end_points()
    values=fn.values_at_end_points()
    n=len(bkpts)
    m=ceil(log(n-1,2))
    bkpts2=bkpts+[1+bkpts[i] for i in range(1,n)]
    values2=values+[values[i] for i in range(1,n)]

    prob=pulp.LpProblem("Deltamin",pulp.LpMinimize)

    xyz=pulp.LpVariable.matrix("xyz",range(3))
    vxyz=pulp.LpVariable.matrix("vxyz",range(3))
    lambda_x=pulp.LpVariable.matrix("lambda_x",range(n),0)
    lambda_y=pulp.LpVariable.matrix("lambda_y",range(n),0)
    lambda_z=pulp.LpVariable.matrix("lambda_z",range(2*n-1),0)
    gamma_x=pulp.LpVariable.matrix("gamma_x",range(2*n),0)
    gamma_y=pulp.LpVariable.matrix("gamma_y",range(2*n),0)
    gamma_z=pulp.LpVariable.matrix("gamma_z",range(4*n-2),0)
    s_x=pulp.LpVariable.matrix("s_x",range(m), 0, 1, pulp.LpInteger)
    s_y=pulp.LpVariable.matrix("s_y",range(m), 0, 1, pulp.LpInteger)
    s_z=pulp.LpVariable.matrix("s_z",range(m+1), 0, 1, pulp.LpInteger)

    prob+=vxyz[0]+vxyz[1]-vxyz[2]

    prob+=pulp.lpSum([lambda_x[i] for i in range(n)])==1
    prob+=pulp.lpSum([lambda_y[i] for i in range(n)])==1
    prob+=pulp.lpSum([lambda_z[i] for i in range(2*n-1)])==1
    prob+=pulp.lpSum([lambda_x[i]*bkpts[i] for i in range(n)])==xyz[0]
    prob+=pulp.lpSum([lambda_y[i]*bkpts[i] for i in range(n)])==xyz[1]
    prob+=pulp.lpSum([lambda_z[i]*bkpts2[i] for i in range(2*n-1)])==xyz[2]
    prob+=pulp.lpSum([lambda_x[i]*values[i] for i in range(n)])==vxyz[0]
    prob+=pulp.lpSum([lambda_y[i]*values[i] for i in range(n)])==vxyz[1]
    prob+=pulp.lpSum([lambda_z[i]*values2[i] for i in range(2*n-1)])==vxyz[2]
    prob+=xyz[0]+xyz[1]==xyz[2]
    for i in range(n):
        prob+=lambda_x[i]==gamma_x[2*i+1]+gamma_x[2*i]
        prob+=lambda_y[i]==gamma_y[2*i+1]+gamma_y[2*i]
    for i in range(2*n-1):
        prob+=lambda_z[i]==gamma_z[2*i+1]+gamma_z[2*i]
    prob+=gamma_x[0]==0
    prob+=gamma_x[2*n-1]==0
    prob+=gamma_y[0]==0
    prob+=gamma_y[2*n-1]==0
    prob+=gamma_z[0]==0
    prob+=gamma_z[4*n-3]==0
    for k in range(m):
        prob+=pulp.lpSum([(gamma_x[2*i-1]+gamma_x[2*i])*int(format(i-1,'0%sb' %m)[k])  for i in range(1,n)])==s_x[k]
        prob+=pulp.lpSum([(gamma_y[2*i-1]+gamma_y[2*i])*int(format(i-1,'0%sb' %m)[k])  for i in range(1,n)])==s_y[k]
    for k in range(m+1):
        prob+=pulp.lpSum([(gamma_z[2*i-1]+gamma_z[2*i])*int(format(i-1,'0%sb' %(m+1))[k])  for i in range(1,2*n-1)])==s_z[k]

    return prob

def generate_mip_of_delta_pi_min_cc(fn,solver='Coin'):
    """
    Generate the Convex Combination mip formulation of computing the minimum of delta pi.
    """
    bkpts=fn.end_points()
    values=fn.values_at_end_points()
    n=len(bkpts)
    bkpts2=bkpts+[1+bkpts[i] for i in range(1,n)]
    values2=values+[values[i] for i in range(1,n)]
    p = MixedIntegerLinearProgram(maximization=False, solver=solver)
    xyz = p.new_variable()
    x,y,z = xyz['x'],xyz['y'],xyz['z']
    vxyz = p.new_variable()
    vx,vy,vz = vxyz['vx'],vxyz['vy'],vxyz['vz']
    lambda_x = p.new_variable(nonnegative=True)
    lambda_y = p.new_variable(nonnegative=True)
    lambda_z = p.new_variable(nonnegative=True)
    b_x=p.new_variable(binary=True)
    b_y=p.new_variable(binary=True)
    b_z=p.new_variable(binary=True)

    p.set_objective(vx+vy-vz)

    p.add_constraint(sum([lambda_x[i]*bkpts[i] for i in range(n)])==x)
    p.add_constraint(sum([lambda_y[i]*bkpts[i] for i in range(n)])==y)
    p.add_constraint(sum([lambda_z[i]*bkpts2[i] for i in range(2*n-1)])==z)
    p.add_constraint(x+y==z)
    p.add_constraint(sum([lambda_x[i]*values[i] for i in range(n)])==vx)
    p.add_constraint(sum([lambda_y[i]*values[i] for i in range(n)])==vy)
    p.add_constraint(sum([lambda_z[i]*values2[i] for i in range(2*n-1)])==vz)
    p.add_constraint(sum([lambda_x[i] for i in range(n)])==1)
    p.add_constraint(sum([lambda_y[i] for i in range(n)])==1)
    p.add_constraint(sum([lambda_z[i] for i in range(2*n-1)])==1)
    p.add_constraint(sum([b_x[i] for i in range(1,n)])==1)
    p.add_constraint(sum([b_y[i] for i in range(1,n)])==1)
    p.add_constraint(sum([b_z[i] for i in range(1,2*n-1)])==1)
    # if b_x[i]=0, the constraint is redundant. if b_x[i]=1, x is in the interval [bkpts[i-1],bkpts[i]], and the constraint forces lambda_x[i-1]+lambda_x[i]=1.
    for i in range(1,n):
        p.add_constraint(lambda_x[i-1]+lambda_x[i]>=b_x[i])
        p.add_constraint(lambda_y[i-1]+lambda_y[i]>=b_y[i])
    for i in range(1,2*n-1):
        p.add_constraint(lambda_z[i-1]+lambda_z[i]>=b_z[i])
    return p

def generate_mip_of_delta_pi_min_mc(fn,solver='Coin'):
    """
    Generate the Multiple Choice mip formulation of computing the minimum of delta pi.
    """
    bkpts=fn.end_points()
    values=fn.values_at_end_points()
    n=len(bkpts)
    bkpts2=bkpts+[1+bkpts[i] for i in range(1,n)]
    values2=values+[values[i] for i in range(1,n)]
    p = MixedIntegerLinearProgram(maximization=False, solver=solver)
    xyz = p.new_variable()
    x,y,z = xyz['x'],xyz['y'],xyz['z']
    vxyz = p.new_variable()
    vx,vy,vz = vxyz['vx'],vxyz['vy'],vxyz['vz']
    lambda_x = p.new_variable(nonnegative=True)
    lambda_y = p.new_variable(nonnegative=True)
    lambda_z = p.new_variable(nonnegative=True)
    b_x=p.new_variable(binary=True)
    b_y=p.new_variable(binary=True)
    b_z=p.new_variable(binary=True)
    gamma_x = p.new_variable(nonnegative=True)
    gamma_y = p.new_variable(nonnegative=True)
    gamma_z = p.new_variable(nonnegative=True)

    p.set_objective(vx+vy-vz)

    p.add_constraint(sum([lambda_x[i]*bkpts[i] for i in range(n)])==x)
    p.add_constraint(sum([lambda_y[i]*bkpts[i] for i in range(n)])==y)
    p.add_constraint(sum([lambda_z[i]*bkpts2[i] for i in range(2*n-1)])==z)
    p.add_constraint(x+y==z)
    p.add_constraint(sum([lambda_x[i]*values[i] for i in range(n)])==vx)
    p.add_constraint(sum([lambda_y[i]*values[i] for i in range(n)])==vy)
    p.add_constraint(sum([lambda_z[i]*values2[i] for i in range(2*n-1)])==vz)
    p.add_constraint(sum([lambda_x[i] for i in range(n)])==1)
    p.add_constraint(sum([lambda_y[i] for i in range(n)])==1)
    p.add_constraint(sum([lambda_z[i] for i in range(2*n-1)])==1)
    p.add_constraint(sum([b_x[i] for i in range(1,n)])==1)
    p.add_constraint(sum([b_y[i] for i in range(1,n)])==1)
    p.add_constraint(sum([b_z[i] for i in range(1,2*n-1)])==1)
    # if b_x[i]=1, x is in the interval [bkpts[i-1],bkpts[i]].
    for i in range(n):
        p.add_constraint(lambda_x[i]==gamma_x[2*i+1]+gamma_x[2*i])
        p.add_constraint(lambda_y[i]==gamma_y[2*i+1]+gamma_y[2*i])
    for i in range(2*n-1):
        p.add_constraint(lambda_z[i]==gamma_z[2*i+1]+gamma_z[2*i])

    for i in range(1,n):
        p.add_constraint(b_x[i]==gamma_x[2*i-1]+gamma_x[2*i])
        p.add_constraint(b_y[i]==gamma_y[2*i-1]+gamma_y[2*i])
    for i in range(1,2*n-1):
        p.add_constraint(b_z[i]==gamma_z[2*i-1]+gamma_z[2*i])

    p.add_constraint(gamma_x[0]==gamma_x[2*n-1]==gamma_y[0]==gamma_y[2*n-1]==gamma_z[0]==gamma_z[4*n-3]==0)
    return p

def generate_mip_of_delta_pi_min_dlog(fn,solver='Coin'):
    """
    Generate the Disaggregated Logarithmic mip formulation of computing the minimum of delta pi.
    """
    bkpts=fn.end_points()
    values=fn.values_at_end_points()
    n=len(bkpts)
    m=ceil(log(n-1,2))
    bkpts2=bkpts+[1+bkpts[i] for i in range(1,n)]
    values2=values+[values[i] for i in range(1,n)]
    p = MixedIntegerLinearProgram(maximization=False, solver=solver)
    xyz = p.new_variable()
    x,y,z = xyz['x'],xyz['y'],xyz['z']
    vxyz = p.new_variable()
    vx,vy,vz = vxyz['vx'],vxyz['vy'],vxyz['vz']
    lambda_x = p.new_variable(nonnegative=True)
    lambda_y = p.new_variable(nonnegative=True)
    lambda_z = p.new_variable(nonnegative=True)
    s_x=p.new_variable(binary=True)
    s_y=p.new_variable(binary=True)
    s_z=p.new_variable(binary=True)
    gamma_x = p.new_variable(nonnegative=True)
    gamma_y = p.new_variable(nonnegative=True)
    gamma_z = p.new_variable(nonnegative=True)

    p.set_objective(vx+vy-vz)

    p.add_constraint(sum([lambda_x[i]*bkpts[i] for i in range(n)])==x)
    p.add_constraint(sum([lambda_y[i]*bkpts[i] for i in range(n)])==y)
    p.add_constraint(sum([lambda_z[i]*bkpts2[i] for i in range(2*n-1)])==z)
    p.add_constraint(x+y==z)
    p.add_constraint(sum([lambda_x[i]*values[i] for i in range(n)])==vx)
    p.add_constraint(sum([lambda_y[i]*values[i] for i in range(n)])==vy)
    p.add_constraint(sum([lambda_z[i]*values2[i] for i in range(2*n-1)])==vz)
    p.add_constraint(sum([lambda_x[i] for i in range(n)])==1)
    p.add_constraint(sum([lambda_y[i] for i in range(n)])==1)
    p.add_constraint(sum([lambda_z[i] for i in range(2*n-1)])==1)

    for i in range(n):
        p.add_constraint(lambda_x[i]==gamma_x[2*i+1]+gamma_x[2*i])
        p.add_constraint(lambda_y[i]==gamma_y[2*i+1]+gamma_y[2*i])
    for i in range(2*n-1):
        p.add_constraint(lambda_z[i]==gamma_z[2*i+1]+gamma_z[2*i])
    p.add_constraint(gamma_x[0]==gamma_x[2*n-1]==gamma_y[0]==gamma_y[2*n-1]==gamma_z[0]==gamma_z[4*n-3]==0)

    for k in range(m):
        p.add_constraint(sum([(gamma_x[2*i-1]+gamma_x[2*i])*int(format(i-1,'0%sb' %m)[k])  for i in range(1,n)])==s_x[k])
        p.add_constraint(sum([(gamma_y[2*i-1]+gamma_y[2*i])*int(format(i-1,'0%sb' %m)[k])  for i in range(1,n)])==s_y[k])
    for k in range(m+1):
        p.add_constraint(sum([(gamma_z[2*i-1]+gamma_z[2*i])*int(format(i-1,'0%sb' %(m+1))[k])  for i in range(1,2*n-1)])==s_z[k])
    return p

def write_performance_file_minimum(readfile_path,readfile_name,writefile_path):
    """
    Read from readfile_name and write time performance of computing the minimum using either the solver, the naive method or fast method.
    """
    bkpts=[]
    values=[]
    with open(readfile_path+readfile_name,mode='r') as readfile:
        function = csv.reader(readfile,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        line_count=0
        for row in function:
            if line_count==0:
                line_count=1
                name,two_epsilon,p_epsilon,method_1,method_2=row
            else:
                bkpt,value=row
                bkpts.append(QQ(bkpt))
                values.append(QQ(value))
        global fn
        fn=piecewise_function_from_breakpoints_and_values(bkpts,values)
    readfile.close()
    with open(writefile_path+'result_'+readfile_name,mode='w') as writefile:
        performance_table = csv.writer(writefile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        performance_table.writerow(['name','two_epsilon','p_epsilon','bkpts','vertices','additive_vertices','min','node_selection','lp_size','time(s)'])
        if method_1=='naive':
            def solve_naive(f):
                global dummy
                dummy=SubadditivityTestTree(gmic())
                dummy.min=minimum_of_delta_pi(f)
            global proc2
            proc2=solve_naive
            t=sage_timeit('proc2(fn)',globals(),number=1,repeat=1,seconds=True)
            if t<600:
                t=sage_timeit('proc2(fn)',globals(),seconds=True)
            m=dummy.min
        elif method_1=='cplex':
            def generate_mip(f):
                global mip
                mip=generate_mip_of_delta_pi_min_dlog(f,solver='cplex')
                return mip
            global proc1
            proc1=generate_mip
            gen_time=sage_timeit('proc1(fn)',globals(),number=1,repeat=1,seconds=True)
            def solve_cplex(p):
                global mi
                mi=p.solve()
                return mi
            global sol1
            sol1=solve_cplex
            sol_time=sage_timeit('sol1(mip)',globals(),number=1,repeat=1,seconds=True)
            if sol_time<600:
                sol_time=sage_timeit('mip.solve()',globals(),seconds=True)
            t=[gen_time,sol_time]
        else:
            lp=int(method2)
            def time_min():
                global T
                T=SubadditivityTestTree(fn)
                T.minimum(max_number_of_bkpts=lp,search_method=method_1,solver='cplex')
            global proc
            proc=time_min
            t=sage_timeit('proc()',globals(),number=1,repeat=1,seconds=True)
            if t<600:
                t=sage_timeit('proc()',globals(),seconds=True)
            mi=T.min
        performance_table.writerow([name,two_epsilon,p_epsilon,len(bkpts),number_of_vertices(fn),number_of_additive_vertices(fn),m,method_1,method_2,t])
    writefile.close()


def write_performance_file_objective(readfile_name,writefile_path,epsilon=-1/100):
    """
    Read from readfile_name and write time performance of checking whether the minimum>=epsilon using either the naive method or fast method.
    """
    bkpts=[]
    values=[]
    with open(readfilename,mode='r') as readfile:
        function = csv.reader(readfile,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        line_count=0
        for row in function:
            if line_count==0:
                line_count=1
                name,two_epsilon,p_epsilon,method_1,method_2=row
            else:
                bkpt,value=row
                bkpts.append(QQ(bkpt))
                values.append(QQ(value))
        fn=piecewise_function_from_breakpoints_and_values(bkpts,values)
    readfile.close()

def generate_test_function_library(readfile_name,writefile_path,perturbation_epsilon_list):
    """
    Store the breakpoints and values of (complicated) functions into the files.
    """
    with open(readfile_name,mode='r') as readfile:
        function_table = csv.reader(readfile,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        line_count=0
        previous_name='dummy'
        for row in function_table:
            if line_count==0:
                line_count=1
                continue
            else:
                name,two_epsilon=row
                if previous_name!=name:
                    k=0
                    previous_name=name
                    distance=1
                two_epsilon=QQ(two_epsilon)
                if two_epsilon>distance:
                    continue
                if name[:5]=='bcdsp':
                    slope_value=int(name[22:])
                    old_fn=bcdsp_arbitrary_slope(k=slope_value)
                elif name[:17]=='gj_2_slope_repeat':
                    m,n=name[18:].split('_')
                    old_fn=gj_2_slope_repeat(m=int(m),n=int(n))
                else:
                    old_fn=eval(name)()
                if two_epsilon!=0:
                    fn=symmetric_2_slope_fill_in(old_fn,two_epsilon)
                    distance=max(abs(v) for v in (old_fn-fn).values_at_end_points())
                else:
                    fn=old_fn
                for p_epsilon in perturbation_epsilon_list:
                    new_fn=function_random_perturbation(fn,p_epsilon)
                    with open(writefile_path+str(name)+'_'+str(k)+'_'+'naive.csv',mode='w') as writefile:
                        function_table = csv.writer(writefile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        function_table.writerow([name,two_epsilon,p_epsilon,'naive','naive'])
                        for i in range(len(new_fn.end_points())):
                            function_table.writerow([new_fn.end_points()[i],new_fn.values_at_end_points()[i]])
                    writefile.close()
                    k=k+1

                    with open(writefile_path+str(name)+'_'+str(k)+'_'+'cplex.csv',mode='w') as writefile:
                        function_table = csv.writer(writefile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        function_table.writerow([name,two_epsilon,p_epsilon,'cplex','cplex'])
                        for i in range(len(new_fn.end_points())):
                            function_table.writerow([new_fn.end_points()[i],new_fn.values_at_end_points()[i]])
                    writefile.close()
                    k=k+1

                    for node_selection in ['BFS','DFS','BB']:
                        for lp_size in [0,20,50,100,10000]:
                            with open(writefile_path+str(name)+'_'+str(k)+'.csv',mode='w') as writefile:
                                function_table = csv.writer(writefile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                function_table.writerow([name,two_epsilon,p_epsilon,node_selection,lp_size])
                                for i in range(len(new_fn.end_points())):
                                    function_table.writerow([new_fn.end_points()[i],new_fn.values_at_end_points()[i]])
                            writefile.close()
                            k=k+1
    readfile.close()

def reproduce_function_from_bkpts_and_values(filename):
    """
    Return a function given by its bkpts and values stored in a csv file.
    """
    bkpts=[]
    values=[]
    with open(filename,mode='r') as readfile:
        function = csv.reader(readfile,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        line_count=0
        for row in function:
            if line_count==0:
                line_count=1
                name,two_epsilon,p_epsilon=row
            else:
                bkpt,value=row
                bkpts.append(QQ(bkpt))
                values.append(QQ(value))
        fn=piecewise_function_from_breakpoints_and_values(bkpts,values)
    readfile.close()
    return name,two_epsilon,p_epsilon,fn

def convert_string_to_list_float(string):
    """
    Convert the string of a list to the actual floating point list.
    """
    return [float(l) for l in string[1:-1].split(", ")]

def convert_string_to_list_QQ(string):
    """
    Convert the string of a list to the actual QQ list.
    """
    return [QQ(l) for l in string[1:-1].split(", ")]

def write_function_table(base_function_list,two_slope_fill_in_epsilon_list,filename):
    """
    Generate a csv file which contains all test functions' information.
    """
    with open(filename, mode='w') as file:
        function_table = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        function_table.writerow(['base_function','two_slope_fill_in_epsilon'])
        for s in base_function_list:
            if s=='bcdsp_arbitrary_slope':
                for k in [str(50*i) for i in range(1,21)]:
                    function_table.writerow([s+'_'+k,0])
            elif s=='gj_2_slope_repeat':
                for first in ['10','100','1000']:
                    for second in ['10','100','1000']:
                        function_table.writerow([s+'_'+first+'_'+second,0])
            else:
                for two_slope_epsilon in two_slope_fill_in_epsilon_list:
                    function_table.writerow([s,two_slope_epsilon])
    file.close()

def write_mip_solving_performance(readfile_name,writefile_name,perturbation_epsilon_list,solver='Coin'):
    """
    Solve the dlog mip formulation of the function stored in readfile_name, and write the performance to writefile_name.
    """
    with open(writefile_name+'_'+solver+'.csv',mode='w') as file:
        performance_table = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        performance_table.writerow(['base_function','two_slope_fill_in_epsilon','perturbation_epsilon','# bkpts','generating mip time (s)','solving time (s)'])
        with open(readfile_name+'.csv',mode='r') as readfile:
            function_table = csv.reader(readfile,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            line_count=0
            for row in function_table:
                if line_count==0:
                    line_count=1
                    continue
                else:
                    name,two_epsilon,bkpts,values,t=row
                    actual_bkpts=convert_stringlist_to_list(bkpts)
                    actual_values=convert_stringlist_to_list(values)
                    fn=piecewise_function_from_breakpoints_and_values(actual_bkpts,actual_values)
                    for pert_epsilon in perturbation_epsilon_list:
                        global new_fn
                        global s
                        s=solver
                        new_fn=function_random_perturbation(fn,pert_epsilon)
                        def generate_mip(f,s):
                            global mip
                            mip=generate_mip_of_delta_pi_min_dlog(f,solver=s)
                            return mip
                        global proc1
                        proc1=generate_mip
                        gen_time=sage_timeit('proc1(new_fn,s)',globals(),number=1,repeat=1,seconds=True)
                        sol_time=sage_timeit('mip.solve()',globals(),number=1,repeat=1,seconds=True)
                        performance_table.writerow([name,two_epsilon,pert_epsilon,len(actual_bkpts),gen_time,sol_time])
        readfile.close()
    file.close()

def measure_T_min(fn,max_number_of_bkpts,search_method,solver='Coin',**kwds):
    global f
    f=fn
    t2=sage_timeit('T=SubadditivityTestTree(f)',globals(),seconds=True)
    def time_min(max_number_of_bkpts=max_number_of_bkpts,search_method=search_method,solver=solver,**kwds):
        global T
        T=SubadditivityTestTree(f)
        T.minimum(max_number_of_bkpts=max_number_of_bkpts,search_method=search_method,solver=solver,**kwds)
    global proc
    proc = time_min
    t1=sage_timeit('proc()',globals(),seconds=True,**kwds)
    return [T.min,T.number_of_nodes(),t1-t2]

def measure_T_is_subadditive(fn,max_number_of_bkpts,search_method,solver='Coin',**kwds):
    global f
    f=fn
    t2=sage_timeit('T=SubadditivityTestTree(f)',globals(),seconds=True)
    def time_limit(max_number_of_bkpts=max_number_of_bkpts,search_method=search_method,solver=solver,**kwds):
        global T
        T=SubadditivityTestTree(f)
        T.is_subadditive(stop_if_fail=True, max_number_of_bkpts=max_number_of_bkpts,search_method=search_method,solver=solver,**kwds)
    global proc
    proc = time_limit
    t1=sage_timeit('proc()',globals(),seconds=True,**kwds)
    return [T.number_of_nodes(),t1-t2]

def function_random_perturbation(fn,epsilon,number_of_bkpts=10):
    """
    Return a random perturbation of the given function fn. Randomly perturb function values at random 10 breakpoints.
    """
    values=fn.values_at_end_points()
    n=len(fn.end_points())
    if epsilon==0 or n<=12:
        return fn
    pert_bkpts=random.sample(range(1, n-1), number_of_bkpts)
    for i in pert_bkpts:
        if random.randint(0,1)==0:
            values[i]+=epsilon
        else:
            values[i]-=epsilon
    return piecewise_function_from_breakpoints_and_values(fn.end_points(), values)
    
def histogram_delta_pi(fn,sampling='vertices',q=5,epsilon=1/10000):
    """
    The histogram of the values of delta pi over given points in the complex.
    """
    values=[]
    if sampling=='vertices':
        bkpts=fn.end_points()
        bkpts2=fn.end_points()[:-1]+[1+bkpt for bkpt in fn.end_points()]
        for x in bkpts:
            for y in bkpts:
                values.append(delta_pi(fn,x,y))
        for z in bkpts2[1:-1]:
            for x in bkpts[1:-1]:
                y=z-x
                if 0<y<1 and y not in bkpts:
                    val=delta_pi(fn,x,y)
                    #symmetry
                    values=values+[val,val]
    elif sampling=='grid':
        for i in range(q+1):
            for j in range(q+1):
                x=i/q
                y=j/q
                values.append(delta_pi(fn,x,y))
    else:
        raise ValueError, "Can't recognize sampling."
    return np.histogram(values,bins=[0,epsilon,1/2,1,3/2,2], density=False)

def number_of_additive_vertices(fn):
    counter=0
    bkpts=fn.end_points()
    bkpts2=fn.end_points()[:-1]+[1+bkpt for bkpt in fn.end_points()]
    for x in bkpts:
        for y in bkpts:
            if delta_pi(fn,x,y)==0:
                counter+=1
    for z in bkpts2[1:-1]:
        for x in bkpts[1:-1]:
            y=z-x
            if 0<y<1 and y not in bkpts and delta_pi(fn,x,y)==0:
                counter+=2
    return counter

def number_of_vertices(fn):
    """
    Return the number of vertices of the complex delta_pi.
    """
    bkpts=fn.end_points()
    bkpts2=fn.end_points()[1:-1]+[1+bkpt for bkpt in fn.end_points()[:-1]]
    counter=len(bkpts)^2
    for z in bkpts2:
        for x in bkpts:
            y=z-x
            if 0<y<1 and y not in bkpts:
                #symmetry
                counter+=2
    return counter

def additive_vertices_ratio(fn):
    return number_of_additive_vertices(fn)/number_of_vertices(fn)

def minimum_of_delta_pi(fn):
    """
    Return the min of delta_pi of fn. (Quatratic complexity)
    """
    global_min=10000
    for x in fn.end_points():
        for y in fn.end_points():
            delta=delta_pi(fn,x,y)
            if delta<global_min:
                global_min=delta
    for z in fn.end_points():
        for x in fn.end_points():
            y=z-x
            delta=delta_pi(fn,x,y)
            if delta<global_min:
                global_min=delta
    for z in fn.end_points():
        for x in fn.end_points():
            z=1+z
            y=z-x
            delta=delta_pi(fn,x,y)
            if delta<global_min:
                global_min=delta
    return global_min

def is_goal_reached(fn,goal=0,stop_if_fail=True,keep_exact_solutions=True):
    """
    Return if delta_pi of fn can reach goal-epsilon. (Quatratic complexity)
    """
    exact_solutions=set()
    superior_solutions=set()
    for x in fn.end_points():
        for y in fn.end_points():
            delta=delta_pi(fn,x,y)
            if keep_exact_solutions and delta==goal:
                exact_solutions.add((x,y))
            if delta<goal:
                superior_solutions.add((x,y))
                if stop_if_fail:
                    return True,superior_solutions
    for z in fn.end_points():
        for x in fn.end_points():
            y=z-x
            delta=delta_pi(fn,x,y)
            if keep_exact_solutions and delta==goal:
                exact_solutions.add((x,y))
            if delta<goal:
                superior_solutions.add((x,y))
                if stop_if_fail:
                    return True,superior_solutions
    for z in fn.end_points():
        for x in fn.end_points():
            z=1+z
            y=z-x
            delta=delta_pi(fn,x,y)
            if keep_exact_solutions and delta==goal:
                exact_solutions.add((x,y))
            if delta<goal:
                superior_solutions.add((x,y))
                if stop_if_fail:
                    return True,superior_solutions
    if len(superior_solutions)>0:
        return True,superior_solutions
    else:
        return False,superior_solutions


    
