from __future__ import print_function
import itertools
import sys
import operator as op
from functools import reduce
import time
import numpy as np
import math

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

def permute(nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    def backtrack(first = 0):
        # if all integers are used up
        if first == n:  
            output.append(nums[:])
        for i in range(first, n):
            # place i-th integer first 
            # in the current permutation
            nums[first], nums[i] = nums[i], nums[first]
            # use next integers to complete the permutations
            backtrack(first + 1)
            # backtrack
            nums[first], nums[i] = nums[i], nums[first]
    
    n = len(nums)
    output = []
    backtrack()
    return output
    
def show_grid(g):
    for row in g:
        print(row)
    print()

def test_mols(g):
    ''' check that all entries in g are unique '''
    size = len(g) ** 2
    a = set()
    for row in g:
        a.update(row)
    return len(a) == size

def mols(n):
    ''' Generate a set of mutually orthogonal latin squares 
        n must be prime
    ''' 
    r = range(n)
    # r = range(1,n+1)

    #Generate each Latin square
    allgrids = []
    for k in range(1, n):
        grid = []
        for i in r:
            row = []
            for j in r:
                a = (k*i + j) % n
                row.append(a)
            grid.append(row)
        allgrids.append(grid)

    for g in allgrids:
        show_grid(g)

    print('- ' * 20 + '\n')
    
    

    #Combine the squares to show their orthoganility
    # m = len(allgrids)
    # for i in range(m):
    #     g0 = allgrids[i]
    #     for j in range(i+1, m):
    #         g1 = allgrids[j]
    #         newgrid = []
    #         for r0, r1 in zip(g0, g1):
    #             newgrid.append(zip(r0, r1))
    #         print(test_mols(newgrid))
    #         show_grid(newgrid)
            
    return allgrids

### Computes the no. of copies of each file that the workers in "workersInd" collectively store
# workersInd: list of worker indices (0-indexed)
# workersInd: list (one element per worker) of lists (one element per file of worker), both are 0-indexed
# r: replication
def countUnionFiles(workersInd, caches, r):
    fileCountHt = {}
    for w in workersInd:
        for i in range(len(caches[w])):
            fileCountHt[caches[w][i]] = fileCountHt.get(caches[w][i], 0) + 1
            
    # print("The union of workers in ", workersInd, " has the following files")
    # print(fileCountHt)
    dam = 0
    for f in fileCountHt.keys():
        if fileCountHt[f] >= (r+1)//2: # the majority of the copies of the file has been distorted
            dam += 1
        # print (str(f) + ":", fileCountHt[f])
        
    # print("# of files in union:", len(fileCountHt))
    # print("# of damaged files:", dam)
            
    return fileCountHt, dam


# Construct the Ramanujan bigraph, do the file assignment, compute damages for some parameters
# based on paper "Deterministic Completion of Rectangular Matrices Using Ramanujan Bigraphs -- II"
def computeDamagesRamanujan(computeDam = False, saveMem = False):
    
    # q should be a prime number
    # l is l in original paper not necessarily the computation load per worker for ByzShield
    # q is q in original paper
    # l,q = 3,4
    # l,q = 4,3
    # l,q = 3,3
    # l,q = 5,7 # Case 1
    l,q = 5,5 # Case 2 
    # l,q = 7,5 # Case 3
    
    # l,q = 3,7 # Case 1
    # l,q = 3,5 # Case 1
    # l,q = 4,5 # Case 1
    # l,q = 3,11 # Case 1
    # l,q = 10,5 # Case 2 
    
    print("--------------------------------------------------------------------------------")
    print("Constructing Ramanujan graph...")
    print("--------------------------------------------------------------------------------")
  
    # cyclic shift permutation matrix
    P = np.zeros((q,q))
    for i in range(1,q+1):
        for j in range(1,q+1):
            P[i-1][j-1] = 1 if j%q == (i-1)%q else 0 
    print(P,"\n")
    
    # verify that P is cyclic shift permutation matrix, i.e, P^T = P^(-1)
    print("The following matrices should be identical")
    print(np.linalg.inv(P))
    print(np.transpose(P), "\n")

    # biadjacency matrix of Ramanujan biregular bipartite graph
    B = np.tile(np.identity(q), l) # first block row of B
    step = 1
    for r in range(q-1):
        curRow = np.identity(q)
        expo = r+1
        for i in range(l-1):
            curRow = np.concatenate((curRow, np.linalg.matrix_power(P, expo)), axis=1)
            expo += step
            
        B = np.concatenate((B, curRow), axis=0)
        step += 1
        
    # also compute the left and right degrees of the bipartite graph, i.e.,
    # the number of files per worker (computation load) and the number of workers per file
    if l < q: # we work with B^T, Ramanujan Case 1
        adj = np.transpose(B)
        dL, dR = q, l
    else: # we work with B, Ramanujan Case 2
        adj = B
        dL, dR = l, q
        
    # no. of workers
    K = np.ma.size(adj, axis=0)
    
    # no. of files
    N = np.ma.size(adj, axis=1)
    
    print("There are K =", K, "workers")
    print("Computation load per worker dL =", dL)
    print("Replication factor per file dR =", dR, "\n")
    
    caches = [] # how worker caches are populated
    
    # (needed for proof), same as "caches" but maps a file to a pair (i,j)
    caches_ij = []
    
    # populate workers
    for i in range(K): # for each worker
        curCache = []
        for j in range(np.ma.size(adj, axis=1)): # for each file
            if adj[i][j] == 1:
                curCache.append(j)
                    
        caches.append(curCache)
        caches_ij.append([(x//dL,x%dL) for x in curCache])
    
    # print caches_ij
    print("TEST: mapping of worker files to tuples used in appendix")
    for i in range(K): # for each worker
        if i < dL:
            print(("U_"+str(i)), caches_ij[i])
        else:
            tmp_ind = i//dL
            # print(tmp_ind)
            print(("U_"+str(i)), caches_ij[i], [(tmp_ind*x[0]-x[1])%dL for x in caches_ij[i]])
    print()
        
        
    # SVD of biadjacency matrix
    _, sigma, _ = np.linalg.svd(adj)
    print("The following values should be identical")
    print(sigma[0])
    print(np.sqrt(q*l), "\n")
    
    # singular values based on theoretical results
    if l < q: # we work with B^T
        print("The singular values of adj=B^T are:" )
        print(sigma, "\n")
        print("The singular values of adj=B^T should be:" )
        print(np.sqrt(q*l), "with multiplicity 1")
        print(np.sqrt(q), "with multiplicity", l*(q-1))
        print("0 with multiplicity", l-1, "\n")
    else: # we work with B
        print("The singular values of adj=B are:" )
        print(sigma, "\n")
        print("The singular values of adj=B should be:" )
        print(np.sqrt(q*l), "with multiplicity 1")
        if l%q == 0:
            print(np.sqrt(l), "with multiplicity", (q-1)*q)
        else:
            k = l%q
            print(np.sqrt(l+q-k), "with multiplicity", (q-1)*k)
            print(np.sqrt(l-k), "with multiplicity", (q-1)*(q-k))
            
        print("0 with multiplicity", q-1, "\n")
        
    
    # Left and right degrees of graph
    dLactual = np.sum(adj, axis=1)
    dRactual = np.sum(adj, axis=0)
    
    print("Left degrees should be identical")
    print(dLactual)
    print("and equal to", dL, "\n")
    
    print("Right degrees should be identical")
    print(dRactual)
    print("and equal to", dR, "\n")
    
    # let Q denote the no. of attacked workers for the sequel
    
    # similar to MOLS case, see for conventions
    # min_Q = (dR+1)//2
    min_Q = 3
    max_Q = 7
    if min_Q < (dR+1)/2 or max_Q > K/2 or min_Q > max_Q: # CHECK max_Q > ... condition 
        sys.exit("Number of Byzantines invalid.")
        
    # See MOLS code
    M_unscaled = np.matmul(adj, np.transpose(adj))
    eigenM_unscaled, _ = np.linalg.eig(M_unscaled)
    eigen_sortedM_unscaled = np.sort(np.real(eigenM_unscaled))[::-1]
    print("The following eigenvalues of M_unscaled should be the squares of the singular values of adj:")
    print(eigen_sortedM_unscaled)
    print("so they should be:")
    print([x**2 for x in sigma], "\n")
    
    # similar to MOLS case, see for conventions
    adj = 1/np.sqrt(dL*dR)*adj
    M = np.matmul(adj, np.transpose(adj))
    # M = np.matmul(np.transpose(adj), adj)
    eigen, _ = np.linalg.eig(M)
    eigen_sorted = np.sort(np.real(eigen))[::-1]
    
    # See MOLS code
    print("The following should be 1/(dL*dR) the eigenvalues of M_unscaled:")
    print(eigen_sorted)
    print("so they should be:")
    print([1/(dL*dR)*x for x in eigen_sortedM_unscaled], "\n")
    
    # fileCountHt, dam = countUnionFiles([0,2,8], caches, dR)
    
    # the expressions below are from the analysis that holds also for MOLS but without substituting \mu_1
    mu_1 = eigen_sorted[1]
    alpha_vec = [(mu_1 + (1-mu_1)*Q/K)*dR for Q in range(min_Q, max_Q+1)]
    beta_vec = [(Q*dL/dR)/(mu_1 + (1-mu_1)*Q/K) for Q in range(min_Q, max_Q+1)]
    gamma_vec = [(Q*dL - (Q*dL/dR)/(mu_1 + (1-mu_1)*Q/K))/((dR-1)/2) for Q in range(min_Q, max_Q+1)]
    delta_vec = [Q*dL/((dR+1)/2) for Q in range(min_Q, max_Q+1)]
    
    # similar to MOLS case, see for conventions
    if computeDam:
        if not saveMem: max_dam_sets = [[] for i in range(min_Q, max_Q+1)]
        max_dam = [0 for i in range(min_Q, max_Q+1)]
        
        for Q in range(min_Q, max_Q+1):
            no_sets = int(ncr(K,Q))
            print("Currently", no_sets, "byzantine sets are examined")
            if not saveMem: dam = [None for i in range(no_sets)]            
            start = time.time()            
            byzantine_sets = itertools.combinations(range(0,K), Q)
            # byzantine_sets = itertools.combinations(range(10,25), q)
            i = 0
            for curByz in byzantine_sets:
                _, curDam = countUnionFiles(curByz, caches, dR)
                max_dam[Q-min_Q] = max(max_dam[Q-min_Q], curDam)
                if not saveMem: dam[i] = curDam
                
                i += 1
            
 
            byzantine_sets = itertools.combinations(range(0,K), Q)
            i = 0
            one_max_dam_set = None
            for curByz in byzantine_sets:
                if not saveMem:
                    curDam = dam[i]
                else:
                    _, curDam = countUnionFiles(curByz, caches, dR)
                if curDam == max_dam[Q-min_Q]:
                    if not saveMem: max_dam_sets[Q-min_Q].append(curByz)
                    
                    if one_max_dam_set is None:
                        one_max_dam_set = curByz
                    
                i += 1
                
            end = time.time()
            print("For Q =", Q, "a set of workers with max damage of", max_dam[Q-min_Q], "files is U_", one_max_dam_set)
            print("Took time", end-start, "sec.\n")
        
        if not saveMem:
            return adj, M, eigen_sorted, alpha_vec, beta_vec, gamma_vec, delta_vec, caches, max_dam, max_dam_sets, dL, dR, min_Q, max_Q, K, N, B, P, sigma
        else:
            return adj, M, eigen_sorted, alpha_vec, beta_vec, gamma_vec, delta_vec, caches, max_dam, dL, dR, min_Q, max_Q, K, N, B, P, sigma
            
    else:
        return adj, M, eigen_sorted, alpha_vec, beta_vec, gamma_vec, delta_vec, caches, dL, dR, min_Q, max_Q, K, N, B, P, sigma  
    

# Construct the MOLS, do the file assignment, compute damages for some parameters
# saveMem: skip some operations to save memory when we compute damages
def computeDamagesMOLS(computeDam = False, saveMem = False):
        
    # order of the Latin squares & replication
    # l, r = 9, 5
    # l, r = 7, 5
    # l, r = 7, 3
    l, r = 5, 3
    # l, r = 3, 2
    
    # no. of workers
    K = r*l
    
    # whether you will use rows and columns
    row_cols_used = False
    
    # min_q = (r+1)//2
    min_q = 2
    max_q = 5
    
    if min_q < (r+1)/2 or max_q > K/2 or min_q > max_q:
        sys.exit("Number of Byzantines invalid.")
    
    print("--------------------------------------------------------------------------------")
    print("Constructing MOLS...")
    print("--------------------------------------------------------------------------------")
    
    allMOLS = mols(l)
    
    # print the workers in each parallel class
    print("The workers in parallel classes are")
    for i in range(r):
        print("U_" + str(i*l) + " ... U_" + str(i*l+l-1))
        
    print('\n' + '- ' * 20 + '\n')
        
    # map the workers to parallel classes (hash table and list)
    worker_class = {}
    class_worker = [[] for i in range(r)]
    for i in range(K):
        worker_class[i] = i//l
        class_worker[i//l].append(i)
        
    # how worker caches are populated
    caches = []
    
    # biadjacency matrix between files and workers for the bipartite graph
    adj = np.zeros((K, l**2))
    
    # row-wise assignment (need to change it to populate adacency matrix)
    # if row_cols_used:
    #     for row in range(l):
    #         curCache = [j for j in range(row*l, (row+1)*l)]
    #         caches.append(curCache)
        
    # column-wise assignment (need to change it to populate adacency matrix)
    # if row_cols_used and r > 1:
    #     for col in range(l):
    #         curCache = [col+l*j for j in range(l)]
    #         caches.append(curCache)
       
    ls = r-2 if row_cols_used else r
    
    print("We will use " + str(ls) + " Latin squares.\n")
    
    # if r > 2:
    if r >= 2:
        # for each Latin square we will populate "order" number of workers
        for lsInd in range(ls):
            for symbol in range(l):
                curCache = []
                curWorker = lsInd*l+symbol
                for i in range(l):
                    for j in range(l):
                        if allMOLS[lsInd][i][j] == symbol:
                            curCache.append(i*l+j)
                            adj[curWorker][i*l+j] = 1
                            
                caches.append(curCache)
             
    # Singular values of the 0-1 biadjacency matrix, not needed for the sequel,
    # just for verification of paper
    # "Deterministic Completion of Rectangular Matrices Using Ramanujan Bigraphs -- II, Theorem 1"
    B = adj
    A = np.concatenate(
                       (np.concatenate((np.zeros((K,K)), B), axis=1),
                       np.concatenate((np.transpose(B), np.zeros((l**2,l**2))), axis=1)),
                       axis=0)
    _, sigma, _ = np.linalg.svd(B)
    print("The following values should match")
    print(sigma[0]) # largest singular value
    print(np.sqrt(l*r)) # square root of the product of left and right degrees of the bipartite graph
    print()
    print("The eigenvalues of A should be +- all singular values of B together with some extra zeros")
    eig_A, _ = np.linalg.eig(A)
    eig_A = np.sort(eig_A.real)[::-1] # A is symmetric so it should have only real eigenvalues
    print(eig_A)
    print(sigma)
    print()
    print("Is it a Ramanujan bigraph?")
    print(sigma[1] <= np.sqrt(l-1)+np.sqrt(r-1))
    print()
    
    # The following is by properties of SVD, just for check
    M_unscaled = np.matmul(adj, np.transpose(adj))
    eigenM_unscaled, _ = np.linalg.eig(M_unscaled)
    eigen_sortedM_unscaled = np.sort(np.real(eigenM_unscaled))[::-1]
    print("The following eigenvalues of M_unscaled should be the squares of the singular values of adj = B:")
    print(eigen_sortedM_unscaled)
    print("so they should be:")
    print([x**2 for x in sigma], "\n")
    
    # normalize the biadjacency matrix
    adj = 1/np.sqrt(l*r)*adj
    
    # the following forms of M (square matrix) should have the same set of nonzero eigenvalues
    M = np.matmul(adj, np.transpose(adj))
    # M = np.matmul(np.transpose(adj), adj)
    
    # eigenvalues of M
    eigen, _ = np.linalg.eig(M)
    eigen_sorted = np.sort(np.real(eigen))[::-1] # assumes that the matrix is real and symmetric
    
    # Based on fundamental property of eigenvalues (multiplication by scalar), just for check
    print("The following should be 1/(l*r) the eigenvalues of M_unscaled:")
    print(eigen_sorted)
    print("so they should be:")
    print([1/(l*r)*x for x in eigen_sortedM_unscaled], "\n")
    
    
    # verify that caches from different parallel classes have one-element intersection
    # for i in range(len(caches)-l):
    #     for j in range(i+l-i%l, len(caches)):
    #         print("Intersection of workers U_" + str(i), "and U_" + str(j), set(caches[i]).intersection(set(caches[j])))
        
    # fileCountHt, dam = countUnionFiles([0,5,10], caches, r)
    # perms = permute([11,12,13,14,16,17,18,19,21])
    
        
    # 2nd largest eigenvalue
    mu_1 = eigen_sorted[1]
    print("The following values should match:")
    print("1/r =", 1/r)
    print("mu_1 =", mu_1, "\n")
    
    # quantities based on paper "Bounds on the Expansion Properties of Tanner Graphs" and my analysis
    alpha_vec = [1+(r-1)*q/(r*l) for q in range(min_q, max_q+1)]
    # alpha_vec = [(mu_1 + (1-mu_1)*q/K)*r for q in range(min_q, max_q+1)]
    beta_vec = [q*l/(1+(r-1)*q/(r*l)) for q in range(min_q, max_q+1)]
    # beta_vec = [(q*l/r)/(mu_1 + (1-mu_1)*q/K) for q in range(min_q, max_q+1)]
    gamma_vec = [(2*q**2/r)/(1+(r-1)*q/(r*l)) for q in range(min_q, max_q+1)]
    # gamma_vec = [(q*l - (q*l/r)/(mu_1 + (1-mu_1)*q/K))/((r+1)/2 - 1) for q in range(min_q, max_q+1)]
    
    # For the argument with \beta to work the average degree of the subfiles \alpha needs to be less than (r+1)/2,

    # vanilla no. of distorted files for our bipartite graph
    delta_vec = [q*l/((r+1)/2) for q in range(min_q, max_q+1)]
    
    # print(gamma_vec)

    if computeDam:
    
        if not saveMem:
            # the indices of the Byzantines sets which all have maximal damage for each value of q we will test
            max_dam_sets = [[] for i in range(min_q, max_q+1)]
            
        # minimum/maximum number of distinct parallel class for to make the maximal damage for each value of q
        # initialize to # of classes + 1
        min_dist_class = [r+1 for i in range(min_q, max_q+1)]
        max_dist_class = [0 for i in range(min_q, max_q+1)]
        
        # maximum number of files you can distort for each value of q
        max_dam = [0 for i in range(min_q, max_q+1)]
        
        for q in range(min_q, max_q+1):
            no_sets = int(ncr(K,q))
            print("Currently", no_sets, "byzantine sets are examined")
            
            # maximum number of files you can distort with current value of q
            # max_dam = 0
            
            # damage of each byzantine set
            if not saveMem: dam = [None for i in range(no_sets)]
            
            start = time.time()
            
            # itertools.combinations() returns an iterator to tuples
            byzantine_sets = itertools.combinations(range(0,K), q)
            # byzantine_sets = itertools.combinations(range(10,25), q)
            i = 0
            for curByz in byzantine_sets:
                _, curDam = countUnionFiles(curByz, caches, r)
                max_dam[q-min_q] = max(max_dam[q-min_q], curDam)
                if not saveMem: dam[i] = curDam
                
                i += 1
                    
                
                # test, should be 32 choose 2 for q = 5, l = 7, r = 5
                # try:
                #     if curByz.index(0) >= 0 and curByz.index(7) >= 0 and curByz.index(14) >= 0:
                #         print("Set with 0,7,14 and q=", q)
                # except ValueError:
                #     pass
            
            # iterator points to the end so we need to reset it
            byzantine_sets = itertools.combinations(range(0,K), q)
            i = 0
            one_max_dam_set = None
            for curByz in byzantine_sets:
                if not saveMem: # re-use saved value
                    curDam = dam[i]
                else: # re-compute damage
                    _, curDam = countUnionFiles(curByz, caches, r)
                if curDam == max_dam[q-min_q]:
                    if not saveMem: max_dam_sets[q-min_q].append(curByz)

                    # just keep a copy of a set with maximal damage
                    if one_max_dam_set is None:
                        one_max_dam_set = curByz
                
                    cur_dist_class = len(set([worker_class[w] for w in curByz]))
                    min_dist_class[q-min_q] = min(min_dist_class[q-min_q], cur_dist_class)
                    max_dist_class[q-min_q] = max(max_dist_class[q-min_q], cur_dist_class)
                    
                i += 1
                
            end = time.time()
                   
            print("For q =", q, "a set of workers with max damage of", max_dam[q-min_q], 
                  "files is U_", one_max_dam_set, "and min # of utilized classes =", min_dist_class[q-min_q])
            print("Took time", end-start, "sec.\n")
            
        
        # return results with damage computation
        if not saveMem:
            return allMOLS, class_worker, adj, M, eigen_sorted, alpha_vec, beta_vec, gamma_vec, delta_vec, caches, max_dam, max_dam_sets, min_dist_class, max_dist_class, l, r, min_q, max_q
        else:
            return allMOLS, class_worker, adj, M, eigen_sorted, alpha_vec, beta_vec, gamma_vec, delta_vec, caches, max_dam, l, r, min_q, max_q
    
    else:
        
        # return results without damage computation
        return allMOLS, class_worker, adj, M, eigen_sorted, alpha_vec, beta_vec, gamma_vec, delta_vec, caches, l, r, min_q, max_q
    
        

# prints all Byzantine worker sets with maximal damage for given q which contain byzantine_set as a subset
# min_q is used for indexing purposes
# e.g. byzantineSupersets([0,7,14], 6, min_q, max_dam_sets)
def byzantineSupersets(byzantine_set, q, min_q, max_dam_sets):
    ctr = 0
    for i in range(len(max_dam_sets[q-min_q])):
        curByz = max_dam_sets[q-min_q][i]
        try:
            for w in byzantine_set:
                # will throw an exception if w is not in the set
                curByz.index(w)
            
            print("Maximal damage set which contains", byzantine_set, "for q=", q, "is", curByz)
            ctr += 1
            
        except ValueError:
            pass
        
    print("# of maximal damage sets which contain", byzantine_set, "for q=", q, "is", ctr)


# For BYZSHIELD AWS code, "args" is used only in AWS code
# Decides file assignment for MOLS scheme
# ...
def mols_groups(args, K, rank):
    group_size = 3 # not for AWS code, group_size == args.group_size to be used in AWS code
    
    # l, r = K//args.group_size, args.group_size # MOLS parameters, to be used in AWS code
    l, r = K//group_size, group_size
    
    allMOLS = mols(l)
    ls = r
    ret_group_dict = {x:[] for x in range(0,l**2)} # list of workers (ranks) for each file
    seeds_dict = {x:[] for x in range(1,K+1)} # list of files for each worker (rank)
    for lsInd in range(ls):
        for symbol in range(l):
            curWorker = lsInd*l+symbol # this starts from zero i.e. it is rank-1
            for i in range(l):
                for j in range(l):
                    if allMOLS[lsInd][i][j] == symbol:
                        seeds_dict[curWorker+1].append(i*l+j)
                        ret_group_dict[i*l+j].append(curWorker+1)
    return ret_group_dict, seeds_dict[rank], list(ret_group_dict.keys())


# For BYZSHIELD AWS code (can also be used above for extra check)
# Checks if positive integer n is prime or not
# https://stackoverflow.com/a/17377939/1467434
def is_prime(n):
    if n == 2:
        return True
    if n % 2 == 0 or n <= 1:
        return False

    sqr = int(n**0.5) + 1

    for divisor in range(3, sqr, 2):
        if n % divisor == 0:
            return False
    return True


# For BYZSHIELD AWS code
# Decides file assignment for Ramanujan schemes based on paper "Deterministic Completion of Rectangular Matrices Using Ramanujan Bigraphs -- II"
# Arguments
# K: no. of workers
# case: 1 or 2, Ramanujan Case
# rank: MPI rank of caller worker
# Returns:
# ret_group_dict: dictionary from file to list of workers (ranks) that have it
# seeds_dict[rank]: list of files for caller worker (rank)
#    OR seeds_dict (returned to PS): dictionary from worker (ranks) to list of files that it has
# ret_group_dict.keys(): list of distinct files
def rama_groups(args, K, case, rank): # world_size == no. of workers here
    group_size = 5 # not for AWS code, group_size == args.group_size to be used in AWS code
    
    # m: parameter m of Ramanujan (l in original paper)
    # s: parameter s of Ramanujan (q in original paper), prime
    # f: no. of files
    
    if case == 1: # Ramanujan Case 1, f == s**2
        m, s = group_size, K//group_size # not for AWS code
        # m, s = args.group_size, K//args.group_size
        assert m < s # Case 1 requirement
    elif case == 2: # Ramanujan Case 2, f == m*s
        rama_m = 10 # not for AWS code
        m, s = rama_m, group_size # not for AWS code
        # m, s = args.rama_m, args.group_size
        assert m >= s and m%s == 0 # Case 2 requirement
        
    assert is_prime(s) # Ramanujan requirement
        
    # cyclic shift permutation matrix
    P = np.zeros((s,s))
    for i in range(1,s+1):
        for j in range(1,s+1):
            P[i-1][j-1] = 1 if j%s == (i-1)%s else 0 

    # biadjacency matrix of Ramanujan biregular bipartite graph
    B = np.tile(np.identity(s), m) # first block row of B
    step = 1
    for r in range(s-1):
        curRow = np.identity(s)
        expo = r+1
        for i in range(m-1):
            curRow = np.concatenate((curRow, np.linalg.matrix_power(P, expo)), axis=1)
            expo += step
            
        B = np.concatenate((B, curRow), axis=0)
        step += 1
        
    # also compute the left and right degrees of the bipartite graph, i.e.,
    # the number of files per worker (computation load) and the number of workers per file
    if m < s: # we work with B^T, Ramanujan Case 1
        adj = np.transpose(B)
    else: # we work with B, Ramanujan Case 2
        adj = B
        
    # no. of files
    f = np.ma.size(adj, axis=1)

    # populate workers
    ret_group_dict = {x:[] for x in range(0,f)} # list of workers (ranks) for each file
    seeds_dict = {x:[] for x in range(1,K+1)} # list of files for each worker (rank)
    for i in range(K): # for each worker
        for j in range(f): # for each file
            if adj[i][j] == 1:
                seeds_dict[i+1].append(j)
                ret_group_dict[j].append(i+1)

    if rank == 0: # PS needs to know both file -> worker and worker -> file assignment to collect and aggregate the gradients
        return ret_group_dict, seeds_dict, [0]*len(ret_group_dict)
    else: # worker
        return ret_group_dict, seeds_dict[rank], list(ret_group_dict.keys())



# just invertes the worker-files dictionary
def cachesWtoWcaches(caches):
    from collections import defaultdict
    fileHt = defaultdict(list)
    for w in range(len(caches)):
        files = caches[w]
        for f in files:
            fileHt[f].append(w+1)
    return fileHt
        

computeDam = True
saveMem = True

if computeDam and not saveMem:
    # MOLS results with damage computation, don't save memory
    allMOLS, class_worker, adj, M, eigen_sorted, alpha_vec, beta_vec, gamma_vec, delta_vec, caches, max_dam, max_dam_sets, min_dist_class, max_dist_class, l, r, min_q, max_q = computeDamagesMOLS(computeDam, saveMem)
elif computeDam and saveMem:
    # MOLS results with damage computation, save memory
    allMOLS, class_worker, adj, M, eigen_sorted, alpha_vec, beta_vec, gamma_vec, delta_vec, caches, max_dam, l, r, min_q, max_q = computeDamagesMOLS(computeDam, saveMem)
else:
    # MOLS results without damage computation
    allMOLS, class_worker, adj, M, eigen_sorted, alpha_vec, beta_vec, gamma_vec, delta_vec, caches, l, r, min_q, max_q = computeDamagesMOLS(computeDam, saveMem)

if computeDam and not saveMem:
    # Ramanujan results with damage computation, don't save memory
    adj, M, eigen_sorted, alpha_vec, beta_vec, gamma_vec, delta_vec, caches, max_dam, max_dam_sets, dL, dR, min_Q, max_Q, K, N, B, P, sigma = computeDamagesRamanujan(computeDam, saveMem)
elif computeDam and saveMem:
    # Ramanujan results with damage computation, save memory
    adj, M, eigen_sorted, alpha_vec, beta_vec, gamma_vec, delta_vec, caches, max_dam, dL, dR, min_Q, max_Q, K, N, B, P, sigma = computeDamagesRamanujan(computeDam, saveMem)
else:
    # Ramanujan results without damage computation
    adj, M, eigen_sorted, alpha_vec, beta_vec, gamma_vec, delta_vec, caches, dL, dR, min_Q, max_Q, K, N, B, P, sigma = computeDamagesRamanujan(computeDam, saveMem)