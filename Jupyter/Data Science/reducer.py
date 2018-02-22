import sys

salesTotal = 0
oldKey = None

for line in sys.stdin :
    data = line.strip().split("\t")
        # store, sales
    if len(data) != 2 :
        continue

    thisKey, thisSale = data
        # thisKey = store, thisSale = sale

    if oldKey and oldKey != thisKey :
        print ("{0}\t{1}".format(oldKey, salesTotal))

        salesTotal = 0

    oldKey = thisKey
    salesTotal += float(thisSale)

    # 하지만 마지막 Key가 출력되지 않았다 !!

if oldKey != None :
    print ("{0}\t{1}".format(oldKey, salesTotal))
