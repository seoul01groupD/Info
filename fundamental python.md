# 입력
- 기본 입력: `input()`, `input().split()` 후자의 경우 공백으로 구분하여 입력을 받음
- `map(int ,input().split())` 정수 여러개를 공백으로 구분하여 입력을 받을 수 있음
- 리스트로 입력받고 싶다면 `list(map(int,input().split()))` 이런식으로 짤 수 있음 
- 예를 들어 1 4 2 입력시 [1,4,2]를 입력받음
> ## 빠른 입력
> * `import sys` 후 `sys.stdin.readline().strip()`
> * 앞의 응용으로 `map(int,sys.stdin.readline().split())` 등도 가능
> * 코딩 할 때는 보통 input으로 짜고 코드의 맨 앞에 `input=sys.stdin.readline`을 넣는 경우가 많음 

# 출력
## 기본 출력
- 기본적으로 출력을 하는 방법은 `print(출력하고 싶은 것)` 이다.
- 기본적으로 기본 함수 인자 `end="\n", sep=" "`가 되어 있으며 각각 출력 후 얼마나 공백을 둘 것인지, 출력 요소 간 공백 간격을 얼마나 할 것인지이다.
- 설정하지 않으면 print()후 줄바꿈, 요소 간 공백은 한칸이지만 문제의 출력에 따라 바꿔야 하는 경우가 있다.
## format형식 
- `"{}{}{}".format(a,b,c)` 는 a,b,c가 순서대로 들어가 출력이 된다. 하지만 중괄호 안에 순서대로 210을 대입시 c,b,a 순서로 들어간다. (기본값이 순서대로고 안에 숫자를 부여하서 순서를 정할 수 있다. )
- `"{:d}".format()` 정수 출력
- `"{:5d}".format()` 정수 출력하되 5칸으로 출력하고 오른쪽끝에 채워짐(정수가 6자리 이상이면 기본 출력과 동일)
- `"{:05d}".format()` 빈 칸을 0으로 채움
- `"{: d}".format()` 기호 위치 한 칸 비움 (양수일 때만) 
- `"{:=+d}".format()` 기호를 맨 앞으로 밈
- `"{:15.3f)".format()` 총 15칸, 소수 뒤 3칸
- `"{:g}".format()` 쓸모없는 소수점 제거 (3.0을 3으로 출력)
- `"{:<5}".format()` 왼쪽으로 붙여서 5칸
- `"{:0>5}".format()`: 오른쪽으로 붙여서 5칸 남은 자리 0 
- ^는 가운데 정렬이고 <,>,^앞에 문자쓰면 그 문자로 빈 공간 채움
## f-string
- 기본적으로 문자열앞에 f를 붙이고 변수가 들어갈 자리에 {}를 넣으면 됨
- `f'나의 이름은 {변수1}입니다. 나이는 {변수2}입니다.'`
> `print("1" "2" "3")`은 123이 출력되고, `print("1","2","3")`은 1 2 3이 출력됨
# 연산
- 사칙연산: +,-,*,/
- /는 실수 나누기, //는 몫, %는 나머지
- 거듭제곱 ** 예를 들어 2의 100승은 `2**100`
# 문자
- 보통 '나 "로 감싸서 생성함
- '를 포함하는 문자열을 만들고 싶다면 "로 전체 문자열을 감싸거나 이스케이프 문자를 사용해야 함
- '나 "앞에 \를 붙이면 됨
- 줄바꿈 \n
- 주석: 한 줄은 #을 앞에 붙이고 여러줄은 다 드래그하여 선택후 Ctrl+/ 누르면 됨
- 위, 아래를 """로 감싸는 방법도 있음
## 문자열 관련 함수 (char이라는 문자열 가정)
| 함수  | 기능 |
| :-------------: | :-------------: |
| `len(char)`  | char의 길이 리턴 |
| `char.count(문자)` | char에 포함된 문자 개수 리턴  |
|`char.find(문자)`| 처음으로 문자가 나오는 index 리턴 없으면 -1 리턴| 
|`char.index(문자)`|첨을으로 문자가 나오는 index 리턴 없으면 오류|
|`char.capitalize()`|char의 첫 문자가 대문자인 문자열 리턴|
|`char.replace("a","b")`|char의 "a"를 모두 "b"로 바꾼 문자열 리턴|
|`char.upper()`|char의 문자들 모두 대문자로 바꾼 문자열 리턴|
|`char.lower()`|char의 문자를 모두 소문자로 바꾼 문자열 리턴|
|`char.isupper()`|char의 문자들이 모두 대문자이면 True 아니면 False 리턴|
|`char.islower()`|char의 문자들이 모두 소문자이면 True 아니면 False 리턴|

> 위의 함수들은 원본 문자열을 바꾸지는 못함

## 아스키 코드
* 문자열을 인코딩한 것(숫자)
* 숫자 => 문자 : `chr(숫자)`
* 문자 => 숫자 : `ord(문자)`

## 진법
### 진수 입력
* `a=input()` 후 `a=int(a,n)` 을 하면 됨
* 예를 들어 a="111" 인데 a=int(a,2)를 하면 a에 7이 할당됨
### 진수 출력
* format에 수를 앞의 괄호에 2,8,16순으로 #b, #o, #x 
* `print("{:#o}".format(10))` 하면 Oo12가 출력됨 (0o숫자=8진수, 0x숫자=16진수)
* \#빼면 0o나 0x생략됨  

# Range함수
* `range(A)` 0~A-1까지 속한 range객체 생성
* `range(A,B)` A~B-1까지 속한 range객체 생성
* `range(A,B,C)` A부터 B-1까지 간격 C인 정수들로 이루어진 range객체 생성
*  `reversed(range())` range를 거꾸로 읽음

# 반복문
## for 반복문
``` python 
for 반복자 in 반복할 수 있는 것:
    내용
```
* 반복 가능한 것: 리스트, 문자, 딕셔너리 등 (딕셔너리의 경우 key값이 반복)
## while 문
``` python
while 불 표현식:
    문장
```

# 제어문
* 반복문 벗어나기
``` python
break
```
* 현재 반복 끝내고 다음 반복
``` python
continue
```

# 리스트
## 기본 리스트 메소드
- 리스트 정의
``` python
리스트명 = [값1, 값2, 값3]
```
- 또다른 리스트 생성
``` python
[표현식 for 반복자 in 반복할 수 있는 것 (+if 조건문)] 
```

- 리스트 요소 호출
``` python 
리스트명[i]
리스트명[-i] # 거꾸로 세서 i번째 칸
```
- 리스트의 +, *
``` python
리스트명*2 # 리스트를 2번 반복한 리스트 리턴
리스트1+리스트2 # 두 리스트를 연결한 리스트 리턴(원본 리스트는 그대로)
```
## 리스트 변화:
### 리스트 요소 변화
- 요소 하나 변화
``` python
리스트명[i]=a # index가 i인 원소를 a로 바꿈(리스트의 길이가 i+1보다 작으면 IndexError)
```
### 리스트 요소 추가 (길이 증가)
- 리스트 요소 추가
``` python
리스트명.insert(i, a) # index가 i인 곳에 a를 삽입 ([i:]가 뒤로 한칸씩 밀림) 
```
- 리스트 맨 뒤에 추가
``` python 
리스트명.append(a) # 리스트 맨 뒤에 a추가 
```
- 리스트에 반복자 추가
``` python 
리스트명.extend(반복자) # 리스트 맨 뒤에 반복자의 원소들 추가 (리스트, 튜플, 딕셔너리 등 추가 가능) 
```
### 리스트 요소 제거 (길이 감소)
- 특정 값 제거:
``` python
리스트명.remove(a) # 처음으로 나오는 a 제거 
```
* 특정 인덱스 제거:
``` python
리스트명.pop(i) # index가 i인 요소 제거, 범위도 지정 가능, i입력 안하면 맨뒤 요소가 제거되고 리턴됨
```
``` python
del 리스트명[i] # index가 i인 요소 제거
```
* 모든 요소 제거:
``` python
리스트명.clear()
```
## 기타 리스트 관련 기법
| 함수  | 기능 |
| :-------------: | :-------------: |
| `sum(리스트명)`  | 리스트를 모두 더한 값 리턴(내부 값이 모두 숫자형이어야 가능) |
|`리스트명.count(a)`|리스트 내의 a값의 개수 리턴|
|`min(리스트명), max(리스트명)`|리스트의 최소/최대 리턴(내부 값이 모두 숫자형이어야 가능)|
|`리스트명.index(a)`|a의 index를 리턴|
|`reversed(리스트명)`|리스트 순서가 바뀐 리스트가 출력(원본 그대로)|
|`값 in 리스트명`|값이 있으면 True, 없으면  False 출력(not in도 가능)|
|`list(enumerate(리스트명))`|[(인덱스,요소),(인덱스,요소)...] 이런식으로 생성|
|`'구분자'.join(리스트명)`|리스트 요소 사이에 구분자를 넣어서 만들어줌 (문자열을 더해야 될 때 이를 활용하는 경우가 많음)|
|`zip(리스트1,리스트2)`|[(리스트1 요소1,리스트2 요소1),(리스트1 요소2, 리스트2, 요소2)...] 이런식으로 된 리스트 생성|
# 정렬
## 기본 정렬 
- `리스트명.sort()` 오름차 순으로 정렬
- `sorted(리스트명)` 오름차순으로 정렬, 원본 리스트 변경 안됨
## 특수 정렬
- `sorted(객체,key=옵션)` 옵션을 기준으로 정렬
- `sorted(객체, key=lambda x: (정렬할 것들), reversed=True/False)`
- 예를 들어 x가 이중 리스트 일때 x의 각 원소의 어떤 요소를 기준으로 정렬할 지 고름
``` python
from operator import itemgetter, attregetter 
sorted(객체, key=itemgetter(인덱스)) # 주어진 인덱스의 원소로 정렬
sorted(객체, key=attregetter(항목)) # 주어진 항목에 대한 정렬- 클래스의 객체에 대한 정렬
```
# 튜플
- 리스트와 달리 변경 불가
- ()로 만듦, 단 원소가 1개인 경우 ,를 붙여 (1,) 식으로 생성해야 함 (그렇지 않으면 정수 1로 인식)
- a,b=b,a 로 쉽게 변수 2개 값 교환 가능

# 딕셔너리
## 딕셔너리 생성법
- 지정법1
``` python
{
    키:값
    키:값
    ...
}
```
- 지정법2
``` python
이름=dict(키=값,키=값 ...)
```
- 지정법3
``` python
리스트명=[[키,값],[키,값],...]
dict(리스트명)
```
## 요소 변화 혹은 생성
`딕셔너리명[키]=값` 키가 기존에 있다면 값으로 대체되고 없었다면 새로운 키:값이 생성됨
`del 딕셔너리명[키]` 해당 키와 값이 삭제됨
`딕셔너리명.clear()` 딕셔너리 내용 지움
## 딕셔너리 관련 함수 
| 함수  | 기능 |
| :-------------: | :-------------: |
| `키 in 딕셔너리명`  | 키가 딕셔너리의 키값에 있다면 True, 아니면 False 리턴 |
|`딕셔너리명.get(키)`|키가 있으면 값 리턴, 아니면 none리턴|
|`딕셔너리명.get(키,문구)`|키가 있으면 값 리턴, 아니면 문구 리턴|
|`딕셔너리명.keys(), 딕셔너리명.values()`|각각 딕셔너리들의 키 값, value값을 리스트로 리턴|
|`딕셔너리명.items()`|dict_items([(키,값)],[(키,값)]...) 리턴|
# 집합
## 지정법
- `set()` 빈 집합 선언
- `{요소들}` 로도 집합 선언 가능
- `set(리스트)` 
## 집합의 기본 연산
| 함수  | 기능 |
| :-------------: | :-------------: |
| `집합1`  | |
# Deque
* `from collections import deque` 후 사용
* `a=deque(리스트명)`으로 정의
* `deque명.appendleft(요소)` 왼쪽에 요소 추가
* `deque명.popleft()` 왼쪽 요소 제거
* 나머지는 리스트와 유사하나 `pop(특정 인덱스)`는 안되고 `del deque명[인덱스]` 는 가능
# Heap 
- 최소힙과 최대힙으로 나뉘며 보통의 heap는 최소힙을 말한다. (최대힙은 위로 갈수록 커지는 것, 최소힙은 위로 갈수록 작아지는 것)
- [참고 블로그](https://reakwon.tistory.com/42)
- 배열을 이진 트리로 저장한다고 생각하면 되며 이진 트리 상에서 부모 노드가 자식 노드보다 항상 작게 유지한다.
## 매커니즘
### 원소 추가
- 맨 끝자리에 원소를 넣고 부모 노드와 비교하여 자신의 자리를 찾아감 (부모 노드가 나보다 커질 때까지 올라감)
### 원소 제거
- 가장 위의 원소를 제거하고 그 자리를 가장 끝의 노드로 채움, 이 노드를 그 다음의 2개 중 작은 것과 교체하고 아래로 계속 내려가며 자신의 자리를 찾아감

## heapq 사용
- `import heapq`를 통해 최소힙 기능을 사용할 수 있음
- 관련 함수 
| 함수  | 기능 |
| :-------------: | :-------------: |
| `heapq.heappush(heap명,item)`  | item을 heap에 추가 |
|`heapq.heappop(heap이름)`|heap에서 가장 작은 원소를 pop 후  리턴. heap이 비어있으면 IndexError|
|`heapq.heapify(리스트명)`|리스트를 heap으로 변환|
|`heapq.nlargest(n,heap명)`|heap내에서 1~n번째로 큰 수가 리스트로 반환됨|

## 최소힙으로 최대힙 구현
``` python
d=[1,3,5,7,8]
max_heap=[]
for item in d:
    heapq.heappush(max_heap,(-item,item))
```
하고 최대힙 접근 시 item[1]로 접근