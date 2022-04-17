
counts = [809,834,477,478,307,122,96,102,324,476]


def find_two_smallest(L):
    '''(list of float) -> tuple of (int,int)
    Return a tiple of the indices of the two smallest values in list L
    >>>find_two_smallest([809,834,477,478,307,122,96,102,324,476])
    '''
    #get minimum item in L
    #find the index of that minimum item
    #remove that item from the list
    #find the index of new minimum item in the list
    #put the smallest item back in the list
    #if necessary, adjust the second index
    #return the two indices
    counts_copy = L[:]
    smallest_index = L.index(min(L))
    L.remove(min(L))
    second_min = min(L)
    second_index = counts_copy.index(second_min)
    return(smallest_index, second_index)



A = [1,2,3,4,5]
B = [2,1,3,4,5]
C = [4,2,1,3,5]
D = [5,1,2,3,4]

print(find_two_smallest(A))
print(find_two_smallest(B))
print(find_two_smallest(C))
print(find_two_smallest(D))
print(find_two_smallest(counts))