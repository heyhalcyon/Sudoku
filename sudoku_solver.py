import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as scs # sparse matrix construction 
import scipy.linalg as scl # linear algebra algorithms
import scipy.optimize as sco # for minimization use
from collections import defaultdict
import math


def selected_new_quiz(result, clue, pos):
    '''
        choose one value at `pos` location 
        from the clue matrix and add it to the result matrix
        pos==1: around topleft corner
        pos==2: around topright corner
        pos==3: around bottomleft corner
        pos==4: around bottomright corner
    '''
    new_clue = clue.copy()
    # create a mask matrix contaning the different entries' positions
    coord = np.nonzero(result != new_clue)   
    pos_len = len(coord[0])
    if pos==1:
        # store the value of from the result at `pos`
        select_clue = result[coord[0][0],coord[1][0]]
        # add the value into the clue 
        new_clue[coord[0][0],coord[1][0]] = select_clue
    if pos==2:
        idx = len(coord[0])//4
        select_clue = result[coord[0][idx],coord[1][idx]]
        new_clue[coord[0][idx],coord[1][idx]] = select_clue 
    if pos==3:
        idx = len(coord[0])//4*3
        select_clue = result[coord[0][idx],coord[1][idx]]
        new_clue[coord[0][idx],coord[1][idx]] = select_clue 
    if pos==4:
        select_clue = result[coord[0][-1],coord[1][-1]]
        new_clue[coord[0][-1],coord[1][-1]] = select_clue
    return new_clue


def clue_recovery(result, clue):
    '''
        add missing clues back to the result matrix if 
        they are not already there
    '''
    mask = result != clue
    return mask * clue + result


def str2mat(string, N=9):
    '''
        convert string to 2d array format
    '''
    N = 9
    mat = np.reshape([int(c) for c in string], (N,N))
    return mat

def mat2str(mat):
    '''
        convert 2d array to 1 flat string
    '''
    string = ''
    for element in mat.reshape(-1):
        string += str(element)
    return string


def dup_val(arr):
    '''
        find the duplicates values in `arr`
    '''
    u, c = np.unique(arr, return_counts=True)
    return u[c>1]

def del_dup(X):
    '''
        delete all duplicated values in rows, columns, and boxes and
        set them=0
    '''
    mat = X.copy()
    # store the location of duplicated values
    location = set()
    M = 3

    # iterate through rows and columns
    for i in range(mat.shape[0]):

        # retreive the duplicated values in columns and store the locations
        dups = dup_val(mat[i,:])
        if dups.size>0:
            for dup in dups:
                for coord in np.nonzero((mat[i,:]==dup))[0]:
                    location.add((i,coord))

        # retreive the duplicated values in rows and store the locations
        dups = dup_val(mat[:,i])
        if dups.size>0:
            for dup in dups:
                for coord in np.nonzero((mat[:,i]==dup))[0]:
                    location.add((coord,i))

    # iterate through 3*3 boxes          
    for i in range(M):
        for j in range(M):
            # retreive the duplicated values in each boxes
            dups = dup_val(mat[i*M:i*M+M,j*M:j*M+M])
            if dups.size>0:
                for dup in dups:
                    coord = np.nonzero(mat[i*M:i*M+M,j*M:j*M+M]==dup)
                    for x, y in zip(coord[0],coord[1]):
                        location.add((x+i*M,y+j*M))
    # set all duplicated values to 0
    for loc in location:
        mat[loc]=0
    return mat



def fixed_constraints(N=9):
    rowC = np.zeros(N)
    rowC[0] =1
    rowR = np.zeros(N)
    rowR[0] =1
    row = scl.toeplitz(rowC, rowR)
    ROW = np.kron(row, np.kron(np.ones((1,N)), np.eye(N)))
    
    colR = np.kron(np.ones((1,N)), rowC)
    col  = scl.toeplitz(rowC, colR)
    COL  = np.kron(col, np.eye(N))
    
    M = int(np.sqrt(N))
    boxC = np.zeros(M)
    boxC[0]=1
    boxR = np.kron(np.ones((1, M)), boxC) 
    box = scl.toeplitz(boxC, boxR)
    box = np.kron(np.eye(M), box)
    BOX = np.kron(box, np.block([np.eye(N), np.eye(N) ,np.eye(N)]))
    
    cell = np.eye(N**2)
    CELL = np.kron(cell, np.ones((1,N)))
    
    return scs.csr_matrix(np.block([[ROW],[COL],[BOX],[CELL]]))


# For the constraint from clues, we extract the nonzeros from the quiz string.
def clue_constraint(m, N=9):
    r, c = np.where(m.T)
    v = np.array([m[c[d],r[d]] for d in range(len(r))])
    
    table = N * c + r
    table = np.block([[table],[v-1]])
    
    # it is faster to use lil_matrix when changing the sparse structure.
    CLUE = scs.lil_matrix((len(table.T), N**3))
    for i in range(len(table.T)):
        CLUE[i,table[0,i]*N + table[1,i]] = 1
    # change back to csr_matrix.
    CLUE = CLUE.tocsr() 
    
    return CLUE


def weighted_solve(quiz, eps=10, L=10, string=False):
    '''
        the main weighted L1 linear programming function
        Arguments: 
            quiz: input 
            eps: parameter for weighted implementation
            L: iteration number for weighted implementation
            string: whether or not the input is str or np.array
    '''
    # tolerance for weighted iteration
    tol = 1e-10
    
    A0 = fixed_constraints()
    if string:
        A1 = clue_constraint(str2mat(quiz))
    else :
        A1 = clue_constraint(quiz)

    # Formulate the matrix A and vector B (B is all ones).
    A = scs.vstack((A0,A1))
    A = A.toarray()
    B = np.ones(A.shape[0])

    # Because rank defficiency. We need to extract effective rank.
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    K = np.sum(s > 1e-12)
    S = np.block([np.diag(s[:K]), np.zeros((K, A.shape[0]-K))])
    A = S@vh
    B = u.T@B
    B = B[:K]

    c = np.block([ np.ones(A.shape[1]), np.ones(A.shape[1]) ])
    G = np.block([[-np.eye(A.shape[1]), np.zeros((A.shape[1], A.shape[1]))],\
                         [np.zeros((A.shape[1], A.shape[1])), -np.eye(A.shape[1])]])
    h = np.zeros(A.shape[1]*2)
    H = np.block([A, -A])
    b = B
    ret = sco.linprog(c, G, h, H, b, method='interior-point', options={'tol':1e-6})
    x0 = ret.x[:A.shape[1]] - ret.x[A.shape[1]:]
    # main weighted implementation
    for i in range(L):
        # calculate the weight from the previous X's
        W = (1/(np.abs(x0)+eps))
        c = np.concatenate((W,W),axis=0)
        ret = sco.linprog(c, G, h, H, b, method='interior-point', options={'tol':1e-6})
        x = ret.x[:A.shape[1]] - ret.x[A.shape[1]:]
        # exit if the X's stop improving
        if np.linalg.norm(x0-x) < tol:
            break
        x0 = x
    z = np.reshape(x, (81, 9))
    result = np.reshape(np.array([np.argmax(d)+1 for d in z]), (9,9) )
    return result

def solver(quiz, eps=20,L=30):
    # fill in possible unique solutions  
    quiz = forward_checking(str(quiz))
    clue = quiz
    # first attempt solving the LP problem
    result = weighted_solve(quiz,eps,L)
    # removing all duplicated values
    result_del = del_dup(result)

    # second attempt solving the LP problem using duplicates deleted matrix as input
    if np.any(result_del==0):
        # incase some clues were deleted by del_dup()
        # add clues back to the matrix if they are missing
        quiz_2 = clue_recovery(result_del,clue)
        # solve the LP problem
        result = weighted_solve(quiz_2,eps,L)
        # removing all duplicated values
        result_del = del_dup(result)

        # third attempting 
        if np.any(result_del==0):
            quiz_2 = clue_recovery(result_del,clue)
            result = weighted_solve(quiz_2,eps,L)
            result_del = del_dup(result)

            # delete all duplicates entries and fill in only one entries to the clue
            # from previous result at the `pos`
            if np.any(result_del==0):
                quiz_3_del = clue_recovery(result_del,clue)
                quiz_3 = selected_new_quiz(quiz_3_del, clue,1)
                result = weighted_solve(quiz_3,eps,L)
                result_del = del_dup(result)

                # same as above but at the second `pos`
                if np.any(result_del==0):
                    quiz_4_del = clue_recovery(result_del,clue)
                    quiz_4 = selected_new_quiz(quiz_4_del, clue, 2)
                    result = weighted_solve(quiz_4,eps,L)
                    result_del = del_dup(result)

                    # same as above but at the third `pos`
                    if np.any(result_del==0):
                        quiz_5_del = clue_recovery(result_del,clue)
                        quiz_5 = selected_new_quiz(quiz_5_del, clue, 3)
                        result = weighted_solve(quiz_5,eps,L)
                        result_del = del_dup(result)

                        # same as above but at the fourth `pos`
                        if np.any(result_del==0):
                            quiz_6_del = clue_recovery(result_del,clue)
                            quiz_6 = selected_new_quiz(quiz_6_del, clue, 4)
                            result = weighted_solve(quiz_6,eps,L)
    return result



def my_block(mygb, N=9):
    '''
        Parameter: mygb: a 2d-array of quiz
                   N = 9 : length of row and column
        Output: return a dictionary whose key is the index of box,
                   value is a list of all spot's values in this box
        There are 9 boxes in total. row*column = 3*3
        e.g. (0,1) representing the box is in first row and the second column
    '''
    dd = defaultdict(list)
    for i in range(N):
        for j in range(N):
            key = (math.floor(i/3), math.floor(j/3))
            dd[key].append(mygb[i][j])
    return dd

def forward_checking(input_quiz , N=9):
    '''
        Parameter: input_quiz: a quiz read from a data set
                   N = 9 : length of row and column
        Output: return a modified quiz with more clue.
    First, change spot with no clue to range = 123456789.
    eliminate the values for the range by given original clues from row, column and box.
    Identifing the box by the function "my_block"
    
    '''

    m = np.reshape([int(c) for c in input_quiz],(N,N))
    
    for i in range(N):
        for j in range(N):
            if m[i][j] == 0:
                m[i][j] = 123456789
               
                for k in m[i]:
                    if k < 10 and k != 0:
                        string = str(m[i][j])
                        if str(k) in string:
                            if len(string) > 1:
                                string = string.replace(str(k), '')
                    
                        m[i][j] = int(string)

                for k in m[:,j]:
                   # print("k",k)
                    if k < 10 and k != 0:
                        string = str(m[i][j])
                        if str(k) in string and len(string) > 1:
                            string = string.replace(str(k), '')

                        m[i][j] = int(string)
    
                dd = my_block(m) #block dict

                ll = dd[(math.floor(i/3), math.floor(j/3))]

    
                for k in ll:         #eliminate block
                    if k < 10 and k != 0:
                        string = str(m[i][j])
                        if str(k) in string and len(string) > 1:
                            string = string.replace(str(k), '')

                        
                        m[i][j] = int(string)

    for i in range(N):
        for j in range(N):
            if m[i][j] > 9:
                m[i][j] = 0
                
    return m


