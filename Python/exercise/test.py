N = int(input(''))
answer =[] 
if (N>=3) :
 if (N%5==0) :
  answer.append (N//5)

 elif ((N%5)%3 == 0) :
  answer.append(N//5 + (N%5)//3)

 elif (N%3==0) :
  answer.append(N//3)

 else :
  for a in range(1,int(N/2)+1) :
   if ((a%5==0) and ((N-a)%3==0)) :
    answer.append( (a//5 +(N-a)//3))
   elif ((a%3==0) and ((N-a)%5==0)) :
    answer.append( (a//3 + (N-a)//5))
 if(answer==[]) :
  print(-1)
 else :
  print(min(answer))

    
