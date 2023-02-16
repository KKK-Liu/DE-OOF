import os
import time

CNT_MAX = 120
NUMBER_OF_ZERO = 4

cnt = 0
while(True):
    time.sleep(1.0)
    number_of_zero = os.popen('nvidia-smi').read().count('0%')
    print('number of zero:{} max number of zero:{} old cnt:{}'.format(number_of_zero, NUMBER_OF_ZERO,cnt),end = '')
    if number_of_zero == NUMBER_OF_ZERO:
        cnt = cnt + 1
    else:
        cnt = 0
    print(' new cnt:{} max cnt:{}'.format(cnt, CNT_MAX))
    if cnt == CNT_MAX:
        print("count enough!")
        break
    
# shutdown
print('shutdown!')
os.popen('shutdown')