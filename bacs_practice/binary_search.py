# def binary_search(L,v):
#     ''' (list, object) -> int
#     return the index of the first occurence of value in L, or return -1 if the value us not in L
#     >>>binary_search([1,3,4,4,5,7,9,10],1)
#     0
#     >>>binary_search([1,3,4,4,5,7,9,10],4)
#     2
#     >>>binary_search([1,3,4,4,5,7,9,10],5)
#     4
#     >>>binary_search([1,3,4,4,5,7,9,10],10)
#     7
#     >>>binary_search([1,3,4,4,5,7,9,10],-3)
#     -1   
#     '''
#     i = 0
#     j = len(L) -1
#     while i != j +1 :  
#         m = (i +j)//2
#         if L[m] < v:
#             i = m + 1
#         else:
#             j = m - 1
#     if 0 <= i < len(L) and L[i]==v:
#         return i
#     else:
#         return -1



# print(binary_search([1,3,4,4,5,7,9,10],1))
# print(binary_search([1,3,4,4,5,7,9,10],4)) 
# print(binary_search([1,3,4,4,5,7,9,10],5)) 
# print(binary_search([1,3,4,4,5,7,9,10],10))
# print(binary_search([1,3,4,4,5,7,9,10],-3))   



import time

def time_it(searchMethod, lst, value):
    t1 = time.perf_counter()
    searchMethod(lst,value)
    t2 = time.perf_counter()
    return (t2-t1)


def binary_search_while(L,V):
    i= 0
    j = len(L)-1
    m = (1 + j)//2
    while i != j+1:
        if L[m] < V:
            i = m +1
        else:
            j = m -1

    #value 가 lst 안에 있을 값의 경우 
    if 0 <= i < len(L) and L[i]== V:
        return i
    else:
        return -1
        
    
L = list(range(10000001))

print(time_it(binary_search_while, L, 100))



        