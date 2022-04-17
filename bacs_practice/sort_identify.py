

def find_two_smallest(L):
    L_copy = L[:]
    L_copy.sort()
    smallest_number = L_copy[0]
    sec_small_number = L_copy[1]
    return(L.index(L_copy[0]) ,L.index(L_copy[1]))


P = [1,3,2,4,5,7,6,8,9]
Q = [152,2747,8473,234,164,685,345,14]

print(find_two_smallest(P))
print(find_two_smallest(Q))