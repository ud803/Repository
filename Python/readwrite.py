# 1. 파일 쓰기
s = " string "
f=file("t.txt", "w")  # 파일 객체 얻기
f.write(s)  # 파일에 쓰기
f.close()   # 스트림 닫기

# 2. 파일 읽기
f = file("t.txt") # 두 번째 인수 생략시 읽기 모드
s = f.read()
print s


# 3. 라인 단위로 파일 읽기

#파일 객체의 반복자 이용하기
f = open("t.txt")
for line in f:
    print line

#readline : 한 번에 한 줄씩
f = open("t.txt")
line = f.readline()
while line:
    print line,
    line = f.readline()

#readlines : 파일 전체를 라인 단위로 끊어 리스트에 저장
f = open("t.txt")
for line in f.readlines():
    print line

#xreadlines : 파일 전체를 한번에 읽지 않고 필요할 때만 읽어서 공급.
f = open ("t.txt")
for line in f.xreadlines():
    print line,


# 4. 라인 단위로 파일 쓰기
lines = ['first line\n', 'second line\n', 'third line\n']
f = open("t1.txt", "w")
f.writelines(lines)

lines = ['first line', 'second line', 'third line']
f = open("t1.txt", "w")
f.write('\n',join(lines))

# 5. 단어의 수 구하기
n = len(open("t.txt").read().split())
print n

# 6. 라인의 수 구하기
len(open("t.txt").readlines())
open("t.txt").read().count('\n')

# 7. 문자의 수 구하기
f = open("t.txt")
len(f.read())
os.path.getsize("t.txt")

# 8. 파일 객체 속성들
'''
file.close()        : 입출력 스트림 닫음
file.read([size])   : 원하는 바이트 수만큼 파일에서 읽는다.
file.readline([size]): 라인 하나를 읽어 들인다.
file.readlines()    : 전체 라인을 readline()을 이용하여 읽음. 리스트에 저장
file.write(str)     : 문자열 str을 파일에 씀
file.tell()         : 파일의 현재 위치 리턴

# 파일 처리 모드
'''
r : 읽기 전용
w : 쓰기 전용
a : 파일 끝에 추가 (쓰기 전용)
r+: 읽고 쓰기
w+: 읽고 쓰기(기존 파일 삭제)
a+: 파일 끝에 추가
b : 이진 파일
'''
