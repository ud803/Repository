a = read.csv(file, header=FALSE, sep=",")
str(a)  # show structure
View(a) # csv는 데이터프레임으로 들어온다.

# Missing Variable을 정해주지 않으면 factor타입이 되어버린다.
c = read.csv("c.csv", T, ",", na.strings=c("No Score"))
str(c)  # factor가 아닌 integer로 나오는 것을 알 수 있다.

# Export to CSV
write.csv(c, file="d.csv", row.names=TRUE, sep=",")

# TXT Files
read.table(file, header=, sep=",")
write.table()

# Import from Excel
install.packages("readxl")
library(readxl) #라이브러리 불러오기

read_excel(path, sheet = 1)
xls1 = read.excel("a.xlsx", sheet=1)


# 복습
# 행 붙이기 (행은 서로 다른 데이터인 데이터프레임)
add_row = data.frame()
rbind(example, add_row)
rbind(example, name = "", gender ="", int, int)
# 열 붙이기 (열은 같은 데이터인 벡터)
cbind()
example$newdata = c()
# 일부 출력
example[c(1,3)]
