from dt import divide, entropy, info_gain, gain_ratio, gini, avg_gini_index, chi_squared_test

attr_vals_list = [['a', 'b', 'c'],  # First attribute takes values 'a', 'b', 'c'.
                  ['x', 'y', 'z'],  # Secon attribute takes values 'x', 'y', 'z'.
                  ['1', '2', '3']]  # The class labels are '1', '2', '3'.
data = [['a', 'x', '1'],
        ['a', 'y', '1'],
        ['b', 'x', '2'],
        ['c', 'x', '1'],
        ['c', 'y', '2']]


def test_buckets(buckets1, buckets2):
    if len(buckets1) != len(buckets2):
        return False
    for bucket1, bucket2 in zip(buckets1, buckets2):
        if set(map(tuple, bucket1)) != set(map(tuple, bucket2)):
            return False
    return True


buckets_divide_0 = [[['a', 'x', '1'], ['a', 'y', '1']], [['b', 'x', '2']], [['c', 'x', '1'], ['c', 'y', '2']]]
buckets_divide_1 = [[['a', 'x', '1'], ['b', 'x', '2'], ['c', 'x', '1']], [['a', 'y', '1'], ['c', 'y', '2']], []]
print('divide_0:', test_buckets(divide(data, 0, attr_vals_list), buckets_divide_0))
print('divide_1:', test_buckets(divide(data, 1, attr_vals_list), buckets_divide_1))

print('entropy:', abs(entropy(data, attr_vals_list) - 0.9709505944546686) < 10 ** -5)

print('info_gain_0:', abs(info_gain(data, 0, attr_vals_list)[0] - 0.5709505944546686) < 10 ** -5)
print('info_gain_0_buckets:', test_buckets(info_gain(data, 0, attr_vals_list)[1], buckets_divide_0))

print('info_gain_1:', abs(info_gain(data, 1, attr_vals_list)[0] - 0.01997309402197489) < 10 ** -5)
print('info_gain_1_buckets:', test_buckets(info_gain(data, 1, attr_vals_list)[1], buckets_divide_1))

print('gain_ratio_0:', abs(gain_ratio(data, 0, attr_vals_list)[0] - 0.37514952012034747) < 10 ** -5)
print('gain_ratio_0_buckets:', test_buckets(gain_ratio(data, 0, attr_vals_list)[1], buckets_divide_0))

print('gain_ratio_1:', abs(gain_ratio(data, 1, attr_vals_list)[0] - 0.020570659450692974) < 10 ** -5)
print('gain_ratio_1_buckets:', test_buckets(gain_ratio(data, 1, attr_vals_list)[1], buckets_divide_1))

print('gini:', abs(gini(data, attr_vals_list) - 0.48) < 10 ** -5)

print('avg_gini_index_0:', abs(avg_gini_index(data, 0, attr_vals_list)[0] - 0.2) < 10 ** -5)
print('avg_gini_index_0_buckets:', test_buckets(avg_gini_index(data, 0, attr_vals_list)[1], buckets_divide_0))

print('avg_gini_index_1:', abs(avg_gini_index(data, 1, attr_vals_list)[0] - 0.4666666666666667) < 10 ** -5)
print('avg_gini_index_1_buckets:', test_buckets(avg_gini_index(data, 1, attr_vals_list)[1], buckets_divide_1))

print('chi_squared_0:', abs(chi_squared_test(data, 0, attr_vals_list)[0] - 2.9166666666666665) < 10 ** -5)
print('chi_squared_0_df:', chi_squared_test(data, 0, attr_vals_list)[1] == 2)

print('chi_squared_1:', abs(chi_squared_test(data, 1, attr_vals_list)[0] - 0.13888888888888884) < 10 ** -5)
print('chi_squared_1_df:', chi_squared_test(data, 1, attr_vals_list)[1] == 1)
