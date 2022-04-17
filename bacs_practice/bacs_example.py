#피보나치 수열이란 첫째 및 둘째 항이 1이며 그 두의 모든 항은 바로 앞 두항의 합인 수열이다. 예를 들어, 처음 여섯 항은 각각 1,1,2,3,5,8 이다. 주어진 숫자만큼의 
#피보나치 수열을 만들어주는 함수를 만들어라

# 함수 이름 : Fibo_Sequence

# print(Fibo_Sequence(20))
# print(Fibo_Sequence(15))
# print(Fibo_Sequence(18))

def Fibo_Sequence(input):
    i=1
    fs_1 = 1
    fs_2 = 1
    while i<= input:
        if i ==1:
            print (1)
        elif i ==2:
            print(1)
        else:
            fs = fs_1 + fs_2
            print(fs)
            fs_2 = fs_1
            fs_1 = fs
        i+= 1


#오늘은 bacs 의 회장 선거날이다. votes라는 투표 list 를 기준으로 각 학생이 몇개의 표를 받았는지 dictionary로 나타내는 함수를 만들어라.
# votes = ["홍용석", "정기성", "안태민", "윤효림", "안태민", "정기성", "정기성", "안태민", "윤효림", "윤효림"]
# vote_counter = {}
# print(vote_counter)

투표 집계하기 문제

votes = ["홍용석", "정기성", "안태민", "윤효림", "안태민", "정기성", "정기성", "안태민", "윤효림", "윤효림"]
vote_counter = {}

for name in votes:
    if name not in vote_counter:
        vote_counter[name]= 1
    else:
        vote_counter[name] += 1

print(vote_counter)


#palindrome은 거꾸로 읽어도 같은 단어를 의미한다. 예를 들어, civic, madam과 같은 단어들이 있다.
#palindrome을 판별하는 함수를 만들어라. 

#함수 이름: is_palindrome
#print(is_palindrome("tomato"))
#print(is_palindrome("otwawto"))
#print(is_palindrome("kayak"))


def is_palindrome(word):
    for left in range (len(word)//2):
        right = len(word)-left-1
        if word[left] != word[right]:
            return False
        return True 

