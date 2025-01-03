def status (actual=10, max=100, steps=10):
    if actual == 0:
        print("|", end="")
        return
    
    if (actual % steps == 0):
        print("-", end="")
    

    if actual == max:
        print("|", end="\n")


import time

for i in range(0, 101, 1):
    status(actual=i, max=100, steps=10)
    pass
    time.sleep(.1)

pass