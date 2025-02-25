# 메모이제이션
- 재귀함수에서 이미 호출한 적이 있는 값은 바로 불러온다. 
- 배열에 함수값들을 넣고 배열에 함수값이 있다면 바로 불러오면 됨 
- (딕셔너리가 더 깔끔 n:f(n)이런 식으로 저장)
- ex. 피보나치수

# 깊이 우선 탐색(DFS)
[참고사이트](https://data-marketing-bk.tistory.com/entry/DFS-%EC%99%84%EB%B2%BD-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-%ED%8C%8C%EC%9D%B4%EC%8D%AC) 
- 각 노드에 연결된 상태가 딕셔너리 형태로 표현되어 있다 가정
- ex.
``` python 
graph['A'] = ['B', 'C']
graph['B'] = ['A', 'D']...
```
## 스택/큐 활용 
### 코드
``` python
def dfs(graph, start_node):
 
    ## 기본은 항상 두개의 리스트를 별도로 관리해주는 것
    need_visited, visited = list(), list()
 
    ## 시작 노드를 지정하기 
    need_visited.append(start_node)
    
    ## 만약 아직도 방문이 필요한 노드가 있다면,
    while need_visited:
 
        ## 그 중에서 가장 마지막 데이터를 추출 (스택 구조의 활용)
        node = need_visited.pop()
        
        ## 만약 그 노드가 방문한 목록에 없다면
        if node not in visited:
 
            ## 방문한 목록에 추가하기 
            visited.append(node)
 
            ## 그 노드에 연결된 노드를 
            need_visited.extend(graph[node])
            
    return visited
```
## 재귀함수 이용
### 코드
``` python
def dfs_recursive(graph, start, visited = []):
## 데이터를 추가하는 명령어 / 재귀가 이루어짐 
    visited.append(start)
 
    for node in graph[start]:
        if node not in visited:
            dfs_recursive(graph, node, visited)
    return visited
```
# 너비 우선 탐색(BFS)
[참고사이트](https://data-marketing-bk.tistory.com/entry/BFS-%EC%99%84%EB%B2%BD-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0-%ED%8C%8C%EC%9D%B4%EC%8D%AC)
- 경로 길이가 짧은 순으로 탐색
- 데이터끼리 연결된 상태가 딕셔너리로 있다 가정
### 코드
``` python 
def bfs(graph, start_node):
    need_visited, visited = [], []
    need_visited.append(start_node)
    
    
    while need_visited:
        node = need_visited[0]
        del need_visited[0]
        
        if node not in visited:
            visited.append(node)
            need_visited.extend(graph[node])
    return visited
```
`deque`를 이용해서도 동일하게 구현 가능

# 이중 우선 순위 큐
입력을 받아서 큐에서 최대값을 삭제할수도, 최소값을 삭제할 수도 있으며 마지막에 최소값, 최대값을 출력할 수 있어야 함
[백준 문제](https://www.acmicpc.net/problem/7662)
## 쉬운 방법(오래 걸림)
1. 최대힙과 최소힙을 구현
2. 입력시 최대힙, 최소힙 모두에 입력
3. 최대값 삭제시 최대힙에서 pop한 후 그 값을 최소힙에서 찾아 뺌
4. 최소값을 삭제할거면 최소힙에서 pop한 후 그 값을 최대힙에서 찾아 뺌
- 일일이 최대힙, 최소힙에서 해당 값을 찾아야 해서 오래 걸린다.

## 최적화된 방법
1. 최소힙, 최대힙, 그리고 입력한 값이 나중에 삭제해야 하는 값인지 아닌지를 저장해 놓는다.
2. 이를 위해 처음에 입력값 k를 받으면 deleted=[True]*k를 만들고 입력시 최대힙, 최소힙에 (-인자,순서), (인자,순서)로 각각 삽입 후 deleted[순서]=False로 바꾼다.
3. 그 후 삭제할 때는 최대값을 삭제하는 경우 최대힙에 원소가 있고 deleted(최대힙[0][1])이 True 인동안 최대힙에서 원소를 제거한다. 
4. 그 후 최대힙에 원소가 있다면 해당 원소를 pop하고 해당순서에 해당하는 delete 인자를 True로 수정한다.
5. 최소값 삭제에 대해서도 3과 4를 동일하게 적용한다.
6. 마지막에 최소힙은 최소힙이 비거나 최소힙의 처음 원소가 delete상에서 False가 될 때까지 pop하고 최대힙 역시 최대힙이 비거나 최대힙의 처음 원소가 delete상에서 False가 될 때까지 pop하면 최대힙의 첫 원소는 최대값을, 최소힙의 처음 원소는 최소값을 저장한 채로 있게 된다.

### 코드
``` python
import heapq
import sys
input=sys.stdin.readline
t=int(input())
for i in range(t):
    min_heap=[]
    max_heap=[]
    heapq.heapify(min_heap)
    heapq.heapify(max_heap)
    k=int(input())
    deleted=[True]*k
    for j in range(k):
        com,n=input().split()
        n=int(n)
        match com:
            case "I":
                heapq.heappush(min_heap,(n,j))
                heapq.heappush(max_heap,(-n,j))
                deleted[j]=False
            case "D":
                if n==1:
                    while max_heap and deleted[max_heap[0][1]]:
                        heapq.heappop(max_heap)
                    if max_heap:
                        deleted[max_heap[0][1]]=True
                        heapq.heappop(max_heap)
                else:
                    while min_heap and deleted[min_heap[0][1]]:
                        heapq.heappop(min_heap)
                    if min_heap:
                        deleted[min_heap[0][1]]=True
                        heapq.heappop(min_heap)
    while min_heap and deleted[min_heap[0][1]]:
        heapq.heappop(min_heap)
    while max_heap and deleted[max_heap[0][1]]:
        heapq.heappop(max_heap)                        
    if min_heap and max_heap:
        print(-max_heap[0][0],min_heap[0][0])
    else:
            print("EMPTY")
```
# 다익스트라 알고리즘(Dijkstra algorithm)
[참고 사이트](https://justkode.kr/algorithm/python-dijkstra/)
- 최단 경로 탐색 알고리즘이다. (하나의 정점에서 다른 모든 정점으로 가는 최단 경로)
1. 출발 노드 설정
2. 출발 노드를 기준으로 각 노드의 최소 비용 저장
3. 방문하지 않은 노드 중에서 가장 비용이 적은 노드 선택
4. 해당 노드를 거쳐서 특정 노드로 가는 경우를 고려하여 최소 비용 갱신
5, 위 과정 반복
이 경우 노드간의 거리가 양수일 때만 해결할 수 있다. (현실의 거리는 모두 양수이므로 가능)
* 노드가 꼭 점이어야할 필요가 없다. 상태이어도 된다  [예시문제](https://www.acmicpc.net/problem/28707) (각 배열의 상태가 곧 노드이다.)

# 벨만-포드 알고리즘(Bellman-Ford Algorithm)
[참고사이트](https://velog.io/@kimdukbae/%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EB%B2%A8%EB%A7%8C-%ED%8F%AC%EB%93%9C-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-Bellman-Ford-Algorithm) 
- 다익스트라와 달리 노드간의 거리가 음수인 경우에도 사용할 수 있다.
1. 출발 노드 설정
2. 최단 거리 테이블 초기화
3. 다음 과정을 V-1번 반복
	1. 모든 간선 E를 하나씩 확인
	2. 각 간선을 거쳐 다른 노드로 가는 비용을 계산하여 최단거리 테이블 갱신
- 만약 3번 과정을 한번 더 수행하여 최단 거리 테이블이 갱신된다면 음수 간선 순환	이 존재하는 것 (v번째 반복했을 때 갱신되면 음수 순환이 있는 것)
- 이 알고리즘의 경우 연결그래프가 아닐 때 한번에 안될 수가 있는데 가상의 점을 생성하고 그 점으로부터 단방향, 길이 0인 도로를 모든 점에 건설하고 그 점에서 시작하면 된다.

# 백트래킹
임의의 집합에서 주어진 기준대로 원소의 순서를 선택하는 문제를 푸는데 적합 
- [예시문제](https://www.acmicpc.net/problem/9663) 이 문제는 유명해서 검색하면 자세한 설명이 나온다.

# 플로이드 워셜
- 그래프에서 가능한 모든 노드 쌍에 대해 최단 거리를 구하는 알고리즘
- 복잡도 n^3
- 시작점-중간점-끝점 3중 for문을 돌리는 알고리즘이다.중간점을 가장 바깥으로 돌려야 한다

# MST(최소 신장 트리) 탐색법
[참고사이트](https://gmlwjd9405.github.io/2018/08/29/algorithm-kruskal-mst.html)
- 간선이 적으면 첫번째 방법이, 많으면 두번째 방법이 유리
## Kruskal MST 알고리즘(그리디 알고리즘)
1. 그래프의 간선들을 가중치의 오름차순으로 정렬
2. 정렬된 간선 리스트 중 순서대로 사이클을 형성하지 않는 간선 선택
(가장 낮은 가중치를 선택하되 사이클을 형성하는 간선을 제외)
3. 해당 간선을 mst집합에 추가
4. 새로 형성된 간선에 의해 사이클이 생성되기 위해서는 그 간선의 양 끝정점이 같은 집합에 속해 있으면 사이클이 형성됨 이를 위해 union find 알고리즘 사용 [참고](https://gmlwjd9405.github.io/2018/08/31/algorithm-union-find.html)
    - union find 알고리즘은 크게 3가지 연산으로 component를 구분
        1. make-set(x): x를 유일한 원소로 하는 새로운 집합 생성
        2. union(x,y): x가 속한 집합과 y가 속한 집합을 합침
        3. find(x): x가 속한 집합의 대푯값 반환
    - 이를 배열과 트리로 구현하는 방법이 있음
        - 배열
            - array[i]:i번 원소가 속한 집합의 번호(루트 노드의 번호)
            - make-set(x): `array[i]=i`
            - union(x,y): 배열의 모든 원소를 순회하면서 y의 집합 번호를 x의 집합 번호로 변경
            - find(x): 한 번만에 x가 속한 집합 번호 찾는다.
        - 트리
            - 보통 트리 형태로 구현함
            - 각각의 부모 노드를 root[]에 저장
            - make-set은 그냥 처음 상태(자기 자신이 자신의 루트)
            - union(x,y)는 x의 최고루트에 y의 최고루트를 붙임
            - find는 최고 위의 노드를 반환(자기 자신이 자신의 root가 될 때까지 올라감)

### 추가적인 최적화하는 방법
1. rank 사용
- 더 깊은 쪽에 얕은 쪽을 붙이면 된다 - 각 집합의 깊이를 rank에 저장
2. 경로 압축
- find의 길이를 줄이는 것으로 `return root[x]=find(root[x])`를 하여 find를 하며 지나간 원소들을 root에 붙여 버린다

## Prim MST 알고리즘
- 시작 정점에서 출발하여 신장트리 집합을 단계적으로 확장하는 방법
    1. 시작 정점만 MST집합에 포함
    2. 앞 단계에서 만들어진 MST집합에 인접한 정점들 중 최소 간선으로 연결된 정점을 선택하여 트리 확장
    3. n-1개까지 반복

# 비트마스크
- 이진수를 사용하는 컴퓨터 연산 방식을 이용하여 정수의 이진수 표현을 자료 구조로 쓰는 기법
- 수행시간이 빠르고 코드가 짧으며 메모리 사용량이 적다
- 비트연산자는 비교 연산자보다 우선순위가 낮다
## 기본연산자
| 연산자 | 기능 |
| :-------------: | :-------------: |
| `&`  | and와 기능이 같다. 비트가 둘다 켜져 있는 경우에만 해당 비트를 켠다  |
|`\|`|or와 동일, 비트가 둘 중 하나라도 켜져있으면 켠다|
|`^`|xor와 동일, 비트가 둘 중 하나만 켜져 있으면 켠다|
|`~`|not과 동일, 비트가 켜져 있는건 끄고, 꺼져 있는건 켠다|
|`<<`, `>>`|비트를 왼쪽 혹은 오른쪽으로 원하는만큼 움직이고 빈자리는 0으로 채운다.|
## 응용
### 비트마스크를 이용한 집합 구현
#### 기본원리
- n개의 비트를 활용하면 전체집합의 원소가 n개인 부분집합들을 표시할 수 있다.
- 소속해 있으면 1, 없으면 0으로 표시
- 즉 `A=0`은 공집합, `A=(1<<n)-1`은 전체 집합
- 원소 추가 `A=A|(1<<k)`
- 원소 삭제 `A=A&~(1<<k)`
- 포함 여부 `if(A&(1<<k))`
- 토글: `A=A^(1<<k)`
#### 두 집합에 대한 연산
- `A|B` 합집합
- `A&B` 교집합
- `A&(~B)`는 차집합
- `A^B` 차집합의 합집합
#### 집합 정보 표현
- 집합 크기
``` python
def bitCount(A):
	if A == 0: 
		return 0
	else:
		 return A%2 + bitCount(A / 2)
```
- 최소 원소 `A&-A`
- 최소 원소 지우기 `A=A&(A-1)`

# CCW 알고리즘
- 기하학에서 사용하는 알고리즘으로 AB, BC가 어느 방향으로 꺾이는지를 알아보는 알고리즘(직선, 반시계, 시계)
- 외적값을 계산하면 된다.

# 투 포인터
- 한 리스트내에서 시작점과 끝점 두 개의 위치를 조정하며 조건에 맞는 경우의 수를 세는 방식
- 보통 시작과 끝은 리스트의 처음으로 잡은 경우가 많고 end가 리스트의 범위를 벗어나면 끝내는 while문으로 작성한다

# Convex Hull(볼록다각형)
- [예시문제](https://www.acmicpc.net/problem/1708)
- 이 경우 가장 왼쪽 최하단의 점은 반드시 포함되어야 함을 이용한다.
1. 위의 조건에 부합하는 점을 잡고 이 점을 기준으로 기울기를 기준으로 정렬한다.(기울기가 같다면 x좌표, y좌표순으로 정렬-x좌표순으로 정렬됨)
2. 스택을 형성하고 초기에는 시작점과 처음 값을 대입한다.
3.  점을 하나씩 빼면서 진행되는데 만약 직선의 방향이 반시계 방향이면 스택에 추가하고 직선이면 직전점을 제외한다.
4. 마지막으로 직선의 방향이 시계방향일 경우 선분의 방향이 반시계방향이 될 때까지 스택에서 pop하면 된다.
5.  마지막에 처음점까지 체크한 후 스택의 길이에서 1을 빼면 그것이 convex hull의 점의 개수이다. 

# LCA, Lowest Common Ancestor (최소 공통 조상)
[참고사이트](https://velog.io/@shiningcastle/%EC%B5%9C%EC%86%8C-%EA%B3%B5%ED%86%B5-%EC%A1%B0%EC%83%81-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98) 
- 트리 구조에서 부모노드를 세대별로 나누어 생각
## 관련 용어와 정의
- 희소 테이블: 배열을 이용해 만드는 dp table로 parent=int[n][21]스케일정도가 적당 (이는 2^21이면 100만까지 커버가 되기 때문에)
- parent: 2차원 배열로 n*21(2^21이 100만정도라서 웬만하면 커버가능)-자신의 2^i번째 부모 정보 저장
- depth: n+1크기로 각 노드의 깊이 저장
- check: n+1크기로 노드의 깊이가 계산되었는지 여부
- graph: 그래프의 정보
- 루트 노드가 안정해져 있으면 임의로 정해도 된다.
- 루트 노드에서 깊이 구하는 함수(루트 노드에서부터 dfs), 전체 부모 관계 설정 함수, 최소 공통조상을 찾는 함수가 필요함
## 최적화
- 가장 기본적인 방법은 각 노드의 부모노드를 모두 저장하는 것이나 이는 비효율적이다.
- 최악의 경우 각 질문에 대해 o(n)이 되어 시간초과가 난다.
- 이를 해결하기 위해 각 노드의 2^i번째 부모 노드만 저장하는 것이 효율적이다.
- 이를 이용하는 대표적인 문제가 [백준 1761번](https://www.acmicpc.net/problem/1761)이다

# KMP 문자열 탐색 알고리즘
[참고사이트](https://bowbowbow.tistory.com/6)
- 일반적으로 문서에서 탐색을 하는 알고리즘이다. 
## pi 배열
- 문자열X가 주어질 때 각 인덱스 i에 대해 X[j]와 X[i-j:i+1]이 되는 최대의 j가 pi[i]이다. (j는 i이하여야 한다.)
- 예를 들어 `X=AABBAA`에서 왼쪽의 AA와 오른쪽의 AA가 같고 AAB와 BAA는 다르므로 pi[5]=2이다.
- 위의 문자열을 기준으로 pi를 계산해보면 pi[0]=0 (는 기본설정) pi[1]=1, pi[2]=0, pi[3]=0, pi[4]=1, pi[5]=2
## 알고리즘
- 이를 바탕으로 탐색을 한 번 성공한 후 다음 단계로 빠르게 넘어갈 수 있다.
-  예를 들어 5번째에서 탐색이 잘 안되었다면 다음 시작 위치를 탐색이 막힌 위치를 기준으로 pi배열을 이용하여 갈 수 있다
- pi배열을 구하는 원리에도 이런 방식이 들어간다.

# 세그먼트 트리
[참고사이트](https://cheon2308.tistory.com/entry/%EC%9E%90%EB%A3%8C%EA%B5%AC%EC%A1%B0-%EC%84%B8%EA%B7%B8%EB%A8%BC%ED%8A%B8-%ED%8A%B8%EB%A6%ACSegment-Tree)
- 여러 개의 데이터가 연속적으로 존재할 때 특정한 범위의 데이터 합을 구하는 방법
## 세팅
1. 최상단 노드: 전체 원소를 더한 값
2. 그 다음 두 노드: 전체 원소의 반을 분할하여 가짐(개수상) 예를 들어 전체 데이터수가   12개면 왼쪽은 앞의 6개의 합 오른쪽은 뒤의 6개의 합(왼쪽에 과반수를 줌)
- 이렇게 되면 부모노드에 2를 곱했을 때 왼쪽 자식 노드가 나타남
- 이를 편하게 하기 위해서 시작 index를 1로 하는 것이 좋음- 새로운 트리의 경우 크기가 기존의 n<=2^m인 최대의 m을 구했다는 가정하에 2^(m+1)이 필요
## 알고리즘
- 트리 구현의 경우 재귀적으로 하게 되는데 구간의 길이가 1보다 크면 2개로 분할하여 더하고 아래로 내려가고 길이가 1이면 그 값을 기록하면 된다.
- 이 트리를 만들었을 때 여러 함수를 구현할 수 있다.
### 구간 합을 구하는 함수
- `sum(start,end,node,left,right)`: start와 end는 시작위치, left, right는 구간합을 구하고자 하는 범위, node는 현재 노드를 의미한다.
- 만약 현재 구간이 노드의 구간을 벗어나면 0을 리턴
- 노드의 구간을 포함하면 트리좌표의 값을 리턴
- 노드의 구간이 요청구간을 포함하면 절반씩 나눠서 재귀
- 겹쳐진 경우 재귀탐색

### 요소 변경
- 특정 위치의 요소를 바꿀 때 트리를 바꾸는 것
- 현재 노드의 포용구간에 업데이트 노드가 없으면 그대로 값 반환 아니라면 변경값만큼 +
- 리프노드가 아니라면 다시 자식 노드 탐색, 리프 노드라면 return

# 강한 연결 요소(Strongly Connected Component)
- [참고 사이트](https://yiyj1030.tistory.com/493)
- [백준 문제](https://www.acmicpc.net/problem/2150)
- 대표적인 코사라주 알고리즘에 대해 다뤄보자.

## 코사라주 알고리즘
### 준비
1) 주어진 방향그래프와 역방향 그래프를 만든다.
2) 정점을 담을 스택 만든다.
### 실행
1) 방향그래프의 임의의 정점부터 dfs를 수행하고 dfs가 끝나는 순서대로 스택에 삽입(탐색이 끝나는 순서대로)
    1) dfs를 수행한 후 아직 방문하지 않는 정점이 있으면 그 정점부터 다시 dfs 수행
    2) 모든 정점을 방문하여 dfs를 완료하여 스택에 모든 정점 담음
2) 스택의 top에서부터 pop를 하며 역방향그래프에서 dfs 수행
    1) 이때 탐색되는 모든 정점을 scc로 묶음
    2) 스택이 비어있을 때까지 진행
    3) 스택의 top에 위치한 정점이 이미 방문했다면 pop만 한다.
