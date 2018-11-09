# Make sure current directory is in path.  
# That's not true while doctesting (sage -t).
if '' not in sys.path:
    sys.path = [''] + sys.path

from igp import *

#import pulp
import pandas as pd
import matplotlib.pyplot as plt
import random
import csv
import time
from sage.misc.sage_timeit import sage_timeit

random.seed(500)

default_function_name_list=['bcdsp_arbitrary_slope','chen_4_slope','drlm_backward_3_slope','gj_forward_3_slope','kzh_7_slope_1','kzh_7_slope_2','kzh_7_slope_3','kzh_7_slope_4','kzh_10_slope_1','kzh_28_slope_1','kzh_28_slope_2']
default_two_slope_fill_in_epsilon_list=[1/(i*10) for i in range(1,11)]
default_perturbation_epsilon_list=[i/100 for i in range(3)]
default_max_number_of_bkpts=[0,10,20,40,100,400,1000,10000,100000]



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

def generate_test_function_library(function_name_list,two_slope_fill_in_epsilon_list,filename):
    """
    Store the breakpoints and values of (complicated) functions into the file filename.csv
    """
    def generating_time(fn,epsilon):
        global new_fn
        new_fn=symmetric_2_slope_fill_in(fn,epsilon)
        return new_fn
    with open(filename+'.csv',mode='w') as file:
        function_table = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        function_table.writerow(['base_function','two_slope_fill_in_epsilon','breakpoints','values','generating time (s)'])
        for name in function_name_list:
            global base_fn
            global new_name
            new_name=name
            base_fn=eval(name)()
            function_table.writerow([name,None,base_fn.end_points(),base_fn.values_at_end_points(),sage_timeit('base_fn=eval(new_name)()',globals(),number=2,repeat=2,seconds=True)])
            for two_slope_fill_in_epsilon in two_slope_fill_in_epsilon_list:
                global epsilon
                global proc
                epsilon=two_slope_fill_in_epsilon
                proc=generating_time
                t=sage_timeit('new_fn=proc(base_fn, epsilon)',globals(),number=1,repeat=1,seconds=True)
                function_table.writerow([name,epsilon,new_fn.end_points(),new_fn.values_at_end_points(),t])
    file.close()

def convert_string_to_list_float(string):
    """
    Convert the string of a list to the actual floating point list.
    """
    return [float(l) for l in string[1:-1].split(",")]

def convert_string_to_list_QQ(string):
    """
    Convert the string of a list to the actual QQ list.
    """
    return [QQ(l) for l in string[1:-1].split(",")]

def write_function_table(base_function_list,two_slope_fill_in_epsilon_list):
    """
    Generate a csv file which contains all test functions' information.
    """
    with open('functions.csv', mode='w') as file:
        function_table = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        function_table.writerow(['base_function','two_slope_fill_in_epsilon'])
        for s in base_function_list:
            for two_slope_epsilon in two_slope_fill_in_epsilon_list:
                if s=='bcdsp_arbitrary_slope':
                    for k in ['5','6','8','9','15','20','30']:
                        function_table.writerow([s+'_'+k,two_slope_epsilon])
                else:
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

def write_performance_table(function_name_list,two_slope_fill_in_epsilon_list,perturbation_epsilon_list):
    with open('performance.csv', mode='w') as file:
        performance_table = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        performance_table.writerow(['base_function', 'two_slope_fill_in_epsilon', 'perturbation_epsilon', '# breakpoints','# slopes','# vertices','# additive vertices','add_v/v ratio','delta pi min','is subadditive','# .min nodes (constant)', '# .min nodes (affine)', '# .min nodes (mixed)','.min n/v ratio (constant)','.min n/v ratio (affine)','.min n/v ratio (mixed)', '# .is_subadditive nodes (constant)', '# .is_subadditive nodes (affine)', '# .is_subadditive nodes (mixed)', '.is_subadditive n/v ratio (constant)', '.is_subadditive n/v ratio (affine)', '.is_subadditive n/v ratio (mixed)','time .min (constant) (s)','time .min (affine) (s)','time .min (mixed) (s)', 'time .is_subadditive (constant) (s)','time .is_subadditive (affine) (s)', 'time .is_subadditive (mixed) (s)','time subadditivity_test (s)'])
        for name in function_name_list:
            global base_fn
            base_fn=eval(name)()
            v=number_of_vertices(base_fn)
            add_v=number_of_additive_vertices(base_fn)
            m,n_m_c,t_m_c=measure_T_min(base_fn,bound="constant",number=2,repeat=2)
            m,n_m_a,t_m_a=measure_T_min(base_fn,bound="affine",number=2,repeat=2)
            m,n_m_m,t_m_m=measure_T_min(base_fn,bound="mixed",number=2,repeat=2)
            is_sub,n_i_c,t_i_c=measure_T_is_subadditive(base_fn,bound="constant",number=2,repeat=2)
            is_sub,n_i_a,t_i_a=measure_T_is_subadditive(base_fn,bound="affine",number=2,repeat=2)
            is_sub,n_i_m,t_i_m=measure_T_is_subadditive(base_fn,bound="mixed",number=2,repeat=2)
            performance_table.writerow([name,None,None,len(base_fn.end_points()),number_of_slopes(base_fn),v,add_v,float(add_v/v),m,is_sub,n_m_c,n_m_a,n_m_m, float(n_m_c/v),float(n_m_a/v),float(n_m_m/v),n_i_c,n_i_a,n_i_m,float(n_i_c/v),float(n_i_a/v),float(n_i_m/v),t_m_c,t_m_a,t_m_m,t_i_c,t_i_a,t_i_m,sage_timeit('subadditivity_test(base_fn,stop_if_fail=True)',globals(),number=2,repeat=2,seconds=True)])
            for fill_in_epsilon in two_slope_fill_in_epsilon_list:
                for perturb_epsilon in perturbation_epsilon_list:
                    global fn
                    fn=test_function_from_two_slope_fill_in_extreme_functions(base_fn,fill_in_epsilon,perturb_epsilon)
                    add_v=number_of_additive_vertices(fn)
                    v=number_of_vertices(fn)
                    m,n_m_c,t_m_c=measure_T_min(fn,bound="constant",number=1,repeat=1)
                    m,n_m_a,t_m_a=measure_T_min(fn,bound="affine",number=1,repeat=1)
                    is_sub,n_i_c,t_i_c=measure_T_is_subadditive(fn,bound="constant",number=1,repeat=1)
                    is_sub,n_i_a,t_i_a=measure_T_is_subadditive(fn,bound="affine",number=1,repeat=1)
                    performance_table.writerow([name,None,None,len(fn.end_points()),number_of_slopes(fn),v,add_v,float(add_v/v),m,is_sub,n_m_c,n_m_a,n_m_m, float(n_m_c/v),float(n_m_a/v),float(n_m_m/v),n_i_c,n_i_a,n_i_m,float(n_i_c/v),float(n_i_a/v),float(n_i_m/v),t_m_c,t_m_a,t_m_m,t_i_c,t_i_a,t_i_m,sage_timeit('subadditivity_test(fn,stop_if_fail=True)',globals(),number=1,repeat=1,seconds=True)])
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

def function_random_perturbation(fn,epsilon,number_of_bkpts_ratio=10):
    """
    Return a random perturbation of the given function fn. Randomly perturb function values at randomly chosen 1/10 breakpoints.
    """
    values=fn.values_at_end_points()
    n=len(fn.end_points())
    if epsilon==0:
        return fn
    bkpts_number=n//number_of_bkpts_ratio
    pert_bkpts=random.sample(range(1, n-1), bkpts_number)
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


    
