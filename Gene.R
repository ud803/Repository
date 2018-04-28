## 0. Data Load / Preprocess

# 데이터 불러오기
seer.df <- read.csv("seerLungTest.csv", header=T, na.strings="")

# 비어있는 행 삭제
seer.df <- seer.df[rowSums(is.na(seer.df))==0, ]

# index 열 삭제, rownames 초기화
seer.df <- seer.df[2:5]
row.names(seer.df) <- 1:nrow(seer.df)

# region, population을 factor로 변경
seer.df$region <- factor(seer.df$region)
seer.df$population <- factor(seer.df$population)

## 1. 지역별, 인구집단별, 성별 환자수에 대한 표를 작성하시오

# 단일 표
table(seer.df$region)
table(seer.df$population)
table(seer.df$gender)

# 3차원 표
table(seer.df$region, seer.df$population, seer.df$gender)


## 2. 지역별, 인구집단별, 성별로 발병 환자수 차이를 알아볼 수 있도록 막대그래프를 그리시오.

for (i in c(3, 4, 2)){
  x11()
  barplot(table(seer.df[i]), xlab= names(seer.df)[i], ylab= "Frequency")
}


## 3. 위의 각 데이터에 대해 발병연령에 대한 평균과 분산을 구한 표를 작성하시오.

multi.fun <- function(x) {
  c(mean = mean(x), var = var(x))
}

for (i in c(3, 4, 2)){
  cat("\n\n\nMean & Var with regard to", names(seer.df)[i], "\n")
  print(sapply(split(seer.df[[1]], seer.df[i]), multi.fun))
}


## 4. 위의 각 데이터에 대해 발병연령 분포를 알아볼 수 있도록 boxplot을 그리시오.

for (i in c(3, 4, 2)){
  x11()
  boxplot(split(seer.df[[1]], seer.df[i]), xlab= names(seer.df)[i])
}


## 5. 지역별, 인구집단별, 성별에 따라 발병연령이 차이가 있는지를 통계 검정을 동반하여 분석하시오.

# Basic Assumptions
# 1. Obtained independently and randomly
  # 조사 과정에서 독립성과 임의성이 지켜졌다고 가정
# 2. Data of each factor are normally distributed
  # 아래 ANOVA에서 qqplot을 통해 확인
# 3. normal populations have a common varaince
  # 3번 분산값의 결과, 분산값이 크게 차이나지 않음


aov_1 <- aov(age.at.diagnosis~region, data=seer.df)
aov_2 <- aov(age.at.diagnosis~population, data=seer.df)
aov_3 <- aov(age.at.diagnosis~gender, data=seer.df)


summary(aov_1)
summary(aov_2)
summary(aov_3)


resid_1 <- aov_1$residuals
resid_2 <- aov_2$residuals
resid_3 <- aov_3$residuals

x11()
qqnorm(resid_1, main="Normal Q-Q Plot of REGION")
qqline(resid_1)
x11()
qqnorm(resid_2, main="Normal Q-Q Plot of POPULATION")
qqline(resid_2)
x11()
qqnorm(resid_3, main="Normal Q-Q Plot of GENDER")
qqline(resid_3)
