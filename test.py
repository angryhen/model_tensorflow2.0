import tensorflow as tf
import numpy as np


a = tf.Variable(0)
@tf.function
def add(iter):
    a.assign_add(next(iter))
    tf.print("a", a)

# iter = iter([0,1,2,3,])
# add(iter)
# add(iter)
# add(iter)
# add(iter)
# with open('1.txt', 'w') as f:
#     f.write('11')

# with open('3.txt', 'a+') as f:
#     f.write('22')
b = 'aaaa%2FB_2020-04-26-10-1_134_1.4.0_5_1577418654_2HBKdsmB6X.zip'
c = 'A_2020-04-26-10-1_134_1.4.0_5_1577418654_2HBKdsmB6X.zip'
d = '_2020-04-26-10-1_134_1.4.0_5_1577418654_2HBKdsmB6X.zip'
print(b.find(c))
if b.find(c) != -1:
    print('yes')

# print(c[1:] in b)
with open('/home/du/Desktop/work_project/automated-testing/log/callback_2020-05-11.txt', 'r') as f:
    lines = f.readlines()
    lines = [line.replace('\r', '').replace('\n', '') for line in lines if
             line.replace('\r', '').replace('\n', '') != '']
fn, result = lines[-1].split(' - ')
print(c[1:] in fn)