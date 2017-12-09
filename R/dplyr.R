## dplyr 패키지 ##

install.packages(c("dplyr", "hflights"))
library(dplyr)
library(hflights)

hf = tbl_df(hflights)



## 1. filter ##
# filter(dataframe, cond1, cond2, cond3-1 | cond3-2)

## 2. arrange ##
# default is ascending order. use desc() otherwise.
# arrange(dataframe, Crit1, Crit2, Crit3)
# arrange(dataframe, Crit1, desc(Crit2))

## 3. select ##
# select(dataframe, col1, col2, col3)
# above is same as select(dataframe, col1:col3)
# select(hf, -(Year:DayOfWeek)) :

## 4. mutate ##
# similar to transform
# mutate(dataframe, newColName = newColFunc, newColName2 = newColName1/2)
# can use the name spontaneously

## 5. summarise ##
# summarise(dataframe, delay = mean(DepDelay, na.rm = TRUE))

## 6. group_by ##
# group_by(dataframe, colName)

planes = group_by(hflights_df, TailNum)
delay = summarise(planes, count=n(), dist=mean(Distance, na.rm=T), delay=mean(ArrDelay, na.rm=T))
delay = filter(delay, count>20, dist<2000)
ggplot(delay, aes(dist, delay)) + geom_point(aes(size=count), alpha=0.5) + geom_smooth() + scale_size_area()


## 7. chain ##
# chain() or %>%
a1 <- group_by(hflights, Year, Month, DayofMonth)
a2 <- select(a1, Year:DayofMonth, ArrDelay, DepDelay)
a3 <- summarise(a2, arr = mean(ArrDelay, na.rm = TRUE), dep = mean(DepDelay, +na.rm = TRUE))
a4 <- filter(a3, arr > 30 | dep > 30)

# chain function can abbreviate above commands as follows
hf %>% group_by(Year, Month, DayofMonth) %>% summarise(arr=mean(ArrDelay, na.rm=T), dep=mean(DepDelay, na.rm=T))%>% filter(arr>30 | dep>30)

hff = hflights %>% filter( TailNum == c(sample(names(table(hflights$TailNum)), 4, replace=F))) %>% select( TailNum, ActualElapsedTime)
