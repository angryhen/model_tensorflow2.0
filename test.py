import tensorflow as tf

a = [[1,2,3], [1,5,6]]
b = [[0,1,2], [1,5,2]]
ans = tf.equal(a,1)
result = tf.where(ans, a, b)
print(result)
