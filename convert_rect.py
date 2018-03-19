import sys

inp = sys.argv[1]
data = open(inp)
dat = data.read()
lines = dat.splitlines()
file = open("groundtruth_rect.txt", "w")
cnt = 0
for line in lines:
    item = line.strip().split()
    x1 = int(item[1])
    x2 = int(item[3])
    y1 = int(item[2])
    y2 = int(item[4])
    w = x2 - x1
    h = y2 - y1
    x = (x1 + x2) // 2
    y = (y1 + y2) // 2
    str = "{},{},{},{}\n".format(x1,y1,w,h)
    file.write(str)
    if cnt!=int(item[5]):
        print('[ERROR]',cnt,item[5])
        print(item)
        break
    cnt+=1


print('!:',cnt)