# barplot(height, .. )

barplot(BOD$demand, names.arg = BOD$Time)

barplot(table(mtcars$cyl))
ggplot(pg_mean, aes(x= group, y= weight)) + geom_bar(stat="identity", fill="lightblue", colour = "black")

# x값을 숫자로 인식
ggplot(BOD, aes(x = Time, y = demand)) + geom_bar(stat = "identity")
ggplot(BOD, aes(x = factor(Time), y = demand)) + geom_bar(stat = "identity")

# 이 둘의 차이는 x값을 연속적으로 보느냐, 아니냐의 차이

# x값은 요인으로 변환
qplot(as.factor(BOD$Time), BOD$demand, geom="bar", stat="identity")
ggplot(BOD, aes(x = factor(Time), y=demand)) + geom_bar(stat = "identity")


# 막대 색상 채우기/테두리 설정하기 (fill : 채우기/ colour(or color) : 테두리)
ggplot(pg_mean, aes(x = group, y = weight)) + geom_bar(stat = "identity", fill = "lightblue", colour = "black")
