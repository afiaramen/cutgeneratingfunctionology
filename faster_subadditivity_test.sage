# Make sure current directory is in path.
# That's not true while doctesting (sage -t).
if '' not in sys.path:
    sys.path = [''] + sys.path

from igp import *
from itertools import chain

import itertools
import numpy as np

class SubadditivityTestTreeNode :

    def __init__(self, fn, level, intervals):
        self.intervals=intervals
        self.level=level
        self.function=fn
        self.bkpts1=self.function.end_points()
        self.bkpts2=self.function.end_points()[:-1]+[1+bkpt for bkpt in self.function.end_points()]
        self.vertices=verts(*intervals)
        self.projs=projections(self.vertices)
        self.I_index=find_possible_branching_bkpts_index(self.bkpts1,self.projs[0])
        self.J_index=find_possible_branching_bkpts_index(self.bkpts1,self.projs[1])
        self.K_index=find_possible_branching_bkpts_index(self.bkpts2,self.projs[2])

    def delta_pi_constant_lower_bound(self):
        alpha_I=fn_constant_bounds(self.function,self.projs[0],lower_bound=True,extended=False)
        alpha_J=fn_constant_bounds(self.function,self.projs[1],lower_bound=True,extended=False)
        beta_K=fn_constant_bounds(self.function,self.projs[2],lower_bound=False,extended=True)
        return alpha_I+alpha_J-beta_K

    def delta_pi_upper_bound(self):
        return min(delta_pi(self.function,v[0],v[1]) for v in self.vertices)

    def branching_direction(self):
        new_I,new_J,new_K=self.projs
        return find_branching_direction(self.function,new_I,new_J,new_K,self.I_index,self.J_index,self.K_index)

    def new_intervals(self):
        if self.branching_direction()=='I':
            new_bkpt=self.bkpts1[floor((self.I_index[0]+self.I_index[1])/2)]
            return [[self.intervals[0][0],new_bkpt],self.intervals[1],self.intervals[2]],[[new_bkpt,self.intervals[0][1]],self.intervals[1],self.intervals[2]]
        elif self.branching_direction()=='J':
            new_bkpt=self.bkpts1[floor((self.J_index[0]+self.J_index[1])/2)]
            return [self.intervals[0],[self.intervals[1][0],new_bkpt],self.intervals[2]],[self.intervals[0],[new_bkpt,self.intervals[1][1]],self.intervals[2]]
        elif self.branching_direction()=='K':
            new_bkpt=self.bkpts2[floor((self.K_index[0]+self.K_index[1])/2)]
            return [self.intervals[0],self.intervals[1],[self.intervals[2][0],new_bkpt]],[self.intervals[0],self.intervals[1],[new_bkpt,self.intervals[2][1]]]
        else:
            return None

    def is_divisible(self):
        if not self.I_index and not self.J_index and not self.K_index:
            return False
        else:
            return True


class SubadditivityTestTree :

    def __init__(self,fn,intervals=[[0,1],[0,1],[0,2]]):
        self.function=fn
        self.intervals=intervals

    def root(self):
        return SubadditivityTestTreeNode(self.function,0,self.intervals)

    def is_subadditive(self):
        return self._is_subadditive

    def height(self):
        return self._height

    def number_of_nodes(self):
        return self._number_of_nodes

    def solve(self):
        """
        Return the min of delta pi over the complex.
        """
        GlobalUpperBound=0
        self._height=0
        self._number_of_nodes=0
        self.additive_vertices=[]
        self.nonadditive_vertices=[]
        self.type1_leafs=[]
        self.type2_leafs=[]
        queue=[self.root()]
        while queue:
            current_node=queue.pop(0)
            self._height=current_node.level
            level=current_node.level+1
            self._number_of_nodes+=1
            upper_bound=current_node.delta_pi_upper_bound()
            #search nonsubadditive vertices
            for v in current_node.vertices:
                if delta_pi(self.function,v[0],v[1])<0:
                    self.nonadditive_vertices.append(v)
            if upper_bound<GlobalUpperBound:
                GlobalUpperBound=upper_bound
            #2 types of leafs
            if not current_node.is_divisible():
                for v in current_node.vertices:
                    if delta_pi(self.function,v[0],v[1])==0:
                        self.additive_vertices.append(v)
                self.type1_leafs.append(current_node)
                continue
            if current_node.delta_pi_constant_lower_bound()>GlobalUpperBound:
                self.type2_leafs.append(current_node)
                continue
            #branching
            intervals1,intervals2=current_node.new_intervals()
            queue.append(SubadditivityTestTreeNode(self.function,level,intervals1))
            queue.append(SubadditivityTestTreeNode(self.function,level,intervals2))
        self.strict_subadditive_faces=[]
        for node in self.type2_leafs:
            self.strict_subadditive_faces.append(Face(node.intervals))
        if self.nonadditive_vertices:
            self._is_subadditive=False
        else:
            self._is_subadditive=True
        self.additive_vertices=set(self.additive_vertices)
        self.nonadditive_vertices=set(self.nonadditive_vertices)
        return GlobalUpperBound

def plot_strict_subadditive_2d_faces(fn,faces):
    p = Graphics()
    p += plot_2d_complex(fn)
    for face in faces:
        p += face.plot(rgbcolor = "red", fill_color = "red")
    return p

def histogram_delta_pi(fn,sampling='vertices'):
    """
    The histogram of the values of delta pi over given points in the complex.
    """
    if sampling=='vertices':
        bkpts=fn.end_points()
        bkpts2=fn.end_points()[:-1]+[1+bkpt for bkpt in fn.end_points()]
        values=[]
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
    else:
        try:
            q=int(sampling)
            for i in range(q+1):
                for j in range(q+1):
                    x=i/q
                    y=j/q
                    values.append(delta_pi(fn,x,y))
        except:
            raise ValueError, "Can't recognize sampling."
    return np.histogram(values,bins=5, density=False)

def values_of_delta_pi_over_grid(fn,q):
    """
    Return a matrix representing the values of delta pi over the grid (1/q)Z*(1/q)Z.
    """
    res=np.zeros((q+1,q+1))
    for i in range(q+1):
        for j in range(q+1):
            res[q-j][i]=delta_pi(fn,i/q,j/q)
    return res

def strategic_delta_pi_min(fn,approximation='constant',norm='one',branching_point_selection='median'):
    """
    Return the minimum of delta_pi.
    """
    f=find_f(fn)
    a=delta_pi_min(fn,[0,f],[0,f],[0,f],approximation=approximation,norm=norm,branching_point_selection=branching_point_selection)
    b=delta_pi_min(fn,[0,1],[0,1],[f,1+f],approximation=approximation,norm=norm,branching_point_selection=branching_point_selection)
    c=delta_pi_min(fn,[f,1],[f,1],[1+f,2],approximation=approximation,norm=norm,branching_point_selection=branching_point_selection)
    return min(a,b,c)

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
 
def find_slope_intercept_trivial(X,Y):
    m=(Y[1]-Y[0])/(X[1]-X[0])
    b=Y[1]-m*X[1] 
    return [m,b]          
                           
def find_best_slope_intercept(X,Y,lower_bound=True,solver='GLPK',norm='one'):
    """
    Find the slope and intercept of the affine lower/upper estimator.
    """
    p = MixedIntegerLinearProgram(maximization=False, solver=solver)
    v = p.new_variable()
    if norm=='one':
        m, b= v['m'], v['b']
        if lower_bound:
            p.set_objective(-m*sum(X)-b*len(X))
            for i in range(len(X)):
                p.add_constraint(m*X[i]+b<=Y[i])
        else:
            p.set_objective(m*sum(X)+b*len(X))
            for i in range(len(X)):
                p.add_constraint(m*X[i]+b>=Y[i])
    elif norm=='inf':
        m, b, z=v['m'], v['b'], v['z']
        p.set_objective(z)
        if lower_bound:
            for i in range(len(X)):
                p.add_constraint(z>=Y[i]-m*X[i]-b>=0)
        else:
            for i in range(len(X)):
                p.add_constraint(z>=m*X[i]+b-Y[i]>=0)
    else:
        raise ValueError, "Can't recognize norm."
    p.solve()
    return [p.get_values(m),p.get_values(b)]

def find_affine_bounds(fn,I,lower_bound=True,extended=False,norm='one'):
    """
    Find the affine lower/upper estimator of the function fn over the closed interval I.
    """
    if not extended:
        bkpts=fn.end_points()
    else:
        bkpts=fn.end_points()[:-1]+[1+bkpt for bkpt in fn.end_points()]
    I_index=find_possible_branching_bkpts_index(bkpts,I)
    X,Y=[I[0],I[1]],[fn(fractional(I[0])),fn(fractional(I[1]))]
    if I_index:
        for i in range(I_index[0], I_index[1]+1):
            X.append(bkpts[i])
            Y.append(fn(fractional(bkpts[i])))
        slope,intercept=find_best_slope_intercept(X,Y,lower_bound=lower_bound,norm=norm)
    else:
        slope,intercept=find_slope_intercept_trivial(X,Y)    
    return slope, intercept

def find_possible_branching_bkpts_index(bkpts,I):
    """
    Return [i,j] such that I[0]<bkpts[k]<I[1] for any i<=k<=j. Return None if empty.
    """
    i=bisect_left(bkpts, I[0])
    j=bisect_left(bkpts, I[1])-1
    if I[0]==bkpts[i]:
        i=i+1
    if i>j:
        return None
    else:
        return [i,j]

def fn_constant_bounds(fn,I,lower_bound=True,extended=False):
    """
    Find the constant lower/upper bound of the function fn over the closed interval I.
    """
    if not extended:
        bkpts=fn.end_points()
    else:
        bkpts=fn.end_points()[:-1]+[1+bkpt for bkpt in fn.end_points()]
    I_index=find_possible_branching_bkpts_index(bkpts,I)
    if lower_bound:
        if not I_index:
            return min(fn(fractional(I[0])), fn(fractional(I[1])))
        else:
            return min(fn(fractional(I[0])), fn(fractional(I[1])), min(fn(fractional(bkpts[i])) for i in range(I_index[0], I_index[1]+1)))
    else:
        if not I_index:
            return max(fn(fractional(I[0])), fn(fractional(I[1])))
        else:
            return max(fn(fractional(I[0])), fn(fractional(I[1])), max(fn(fractional(bkpts[i])) for i in range(I_index[0], I_index[1]+1)))

def find_branching_bkpt(fn,I_index,extended=False,branching_point_selection='median'):
    """
    Select the branching point.
    """
    if extended:
        bkpts=fn.end_points()[:-1]+[1+bkpt for bkpt in fn.end_points()]
    else:
        bkpts=fn.end_points()
    if branching_point_selection=='median':
        return bkpts[floor((I_index[0]+I_index[1])/2)]
    elif branching_point_selection=='sparse':
        distance=0
        for k in range(I_index[0],I_index[1]+1):
        # choose large interval. Could be more flexible?
            new_distance=bkpts[k+1]-bkpts[k-1]
            if new_distance>distance:
                distance=new_distance
                best_bkpt=bkpts[k]
        return best_bkpt
    else:
        raise ValueError, "Can't recognize branching_point_selection."

def find_branching_direction(fn,I,J,K,I_index,J_index,K_index):
    lenI=I[1]-I[0]
    lenJ=J[1]-J[0]
    lenK=K[1]-K[0]
    candidates=[]
    if I_index:
        candidates.append(['I',lenI])
    if J_index:
        candidates.append(['J',lenJ])
    if K_index:
        candidates.append(['K',lenK])
    if not candidates:
        return None
    else:
        ind=np.argmax([candidate[1] for candidate in candidates])
        return candidates[ind][0]

def delta_pi_min(fn,I,J,K,approximation='constant',norm='one',branching_point_selection='median'):
    """
    Return the minimum of the function delta fn in the region whose projections are I,J,K.
    """
    bkpts1=fn.end_points()
    bkpts2=fn.end_points()[:-1]+[1+bkpt for bkpt in fn.end_points()]
    vertices=verts(I,J,K)
    if len(vertices)==0:
        return 100
    elif len(vertices)==1:
        return delta_pi(fn,vertices[0][0],vertices[0][1])
    new_I,new_J,new_K=projections(vertices)
    I_index=find_possible_branching_bkpts_index(bkpts1,new_I)
    J_index=find_possible_branching_bkpts_index(bkpts1,new_J)
    K_index=find_possible_branching_bkpts_index(bkpts2,new_K)
    if approximation=='constant':
        alpha_I=fn_constant_bounds(fn,new_I,lower_bound=True,extended=False)
        alpha_J=fn_constant_bounds(fn,new_J,lower_bound=True,extended=False)
        beta_K=fn_constant_bounds(fn,new_K,lower_bound=False,extended=True)
        if alpha_I+alpha_J-beta_K>0:
            return alpha_I+alpha_J-beta_K
    elif approximation=='affine':
        m_I, b_I=find_affine_bounds(fn,new_I,lower_bound=True,extended=False,norm=norm)
        m_J, b_J=find_affine_bounds(fn,new_J,lower_bound=True,extended=False,norm=norm)
        m_K, b_K=find_affine_bounds(fn,new_K,lower_bound=False,extended=True,norm=norm)
        delta_lower_bound=min((m_I*vertex[0]+b_I)+(m_J*vertex[1]+b_J)-(m_K*fractional(vertex[0]+vertex[1])+b_K) for vertex in vertices)
        if delta_lower_bound>0:
            return delta_lower_bound
    else:
        raise ValueError, "Can't recognize approximation."
    upper_bound=min(delta_pi(fn,v[0],v[1]) for v in vertices)
    if upper_bound<0:
        return upper_bound
    branch_direction=find_branching_direction(fn,new_I,new_J,new_K,I_index,J_index,K_index)
    if not branch_direction:
        return upper_bound
    elif branch_direction=='I':
        bkpt=find_branching_bkpt(fn,I_index,extended=False,branching_point_selection=branching_point_selection)
        return min(delta_pi_min(fn,[I[0],bkpt],J,K,approximation=approximation,norm=norm,branching_point_selection=branching_point_selection),delta_pi_min(fn,[bkpt,I[1]],J,K,approximation=approximation,norm=norm,branching_point_selection=branching_point_selection))
    elif branch_direction=='J':
        bkpt=find_branching_bkpt(fn,J_index,extended=False,branching_point_selection=branching_point_selection)
        return min(delta_pi_min(fn,I,[J[0],bkpt],K,approximation=approximation,norm=norm,branching_point_selection=branching_point_selection),delta_pi_min(fn,I,[bkpt,J[1]],K,approximation=approximation,norm=norm,branching_point_selection=branching_point_selection))
    elif branch_direction=='K':
        bkpt=find_branching_bkpt(fn,K_index,extended=True,branching_point_selection=branching_point_selection)
        return min(delta_pi_min(fn,I,J,[K[0],bkpt],approximation=approximation,norm=norm,branching_point_selection=branching_point_selection),delta_pi_min(fn,I,J,[bkpt,K[1]],approximation=approximation,norm=norm,branching_point_selection=branching_point_selection))
    
