
corr=list()
for (j in seq(1:1999)) {
	
word = paste0('V',j)
temp=c()

for (i in seq(j+1,2000)) {
	word2 = paste0('V',i)

	temp = c(temp, cor(a[word],a[word2])[1])

}
corr[[j]] = temp
}
