import pandas as pd
import numpy as np

data = pd.read_csv(
    r'C:\Users\Lenovo\Desktop\Term 2-2565\Data Mining - 261448\Hw3 NN\breast-cancer.data', header=None)

data.columns = ['Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig', 'breast',
                'breast-quad', 'irradiat']

data = data.replace('?', np.NaN)
print('Number of instances = %d' % (data.shape[0]))
print('Number of attributes = %d' % (data.shape[1]))

print('Number of missing values:')
for col in data.columns:
    print('\t%s: %d' % (col, data[col].isna().sum()))

print('Number of rows in original data = %d' % (data.shape[0]))
data2 = data.dropna()
print('Number of rows after discarding missing values = %d' % (data2.shape[0]))


for i in range(0, 277):
    data2['Class'].values[i] = data2['Class'].replace(
        'no-recurrence-events', '1').values[i]
    data2['Class'].values[i] = data2['Class'].replace(
        'recurrence-events', '2').values[i]

    data2['age'].values[i] = data2['age'].replace('10-19', '1').values[i]
    data2['age'].values[i] = data2['age'].replace('20-29', '2').values[i]
    data2['age'].values[i] = data2['age'].replace('30-39', '3').values[i]

    data2['age'].values[i] = data2['age'].replace('40-49', '4').values[i]
    data2['age'].values[i] = data2['age'].replace('50-59', '5').values[i]
    data2['age'].values[i] = data2['age'].replace('60-69', '6').values[i]

    data2['age'].values[i] = data2['age'].replace('70-79', '7').values[i]
    data2['age'].values[i] = data2['age'].replace('80-89', '8').values[i]
    data2['age'].values[i] = data2['age'].replace('90-99', '9').values[i]

    data2['menopause'].values[i] = data2['menopause'].replace(
        'lt40', '1').values[i]
    data2['menopause'].values[i] = data2['menopause'].replace(
        'ge40', '2').values[i]
    data2['menopause'].values[i] = data2['menopause'].replace(
        'premeno', '3').values[i]

    data2['tumor-size'].values[i] = data2['tumor-size'].replace(
        '0-4', '1').values[i]
    data2['tumor-size'].values[i] = data2['tumor-size'].replace(
        '5-9', '2').values[i]
    data2['tumor-size'].values[i] = data2['tumor-size'].replace(
        '10-14', '3').values[i]
    data2['tumor-size'].values[i] = data2['tumor-size'].replace(
        '15-19', '4').values[i]

    data2['tumor-size'].values[i] = data2['tumor-size'].replace(
        '20-24', '5').values[i]
    data2['tumor-size'].values[i] = data2['tumor-size'].replace(
        '25-29', '6').values[i]
    data2['tumor-size'].values[i] = data2['tumor-size'].replace(
        '30-34', '7').values[i]
    data2['tumor-size'].values[i] = data2['tumor-size'].replace(
        '35-39', '8').values[i]

    data2['tumor-size'].values[i] = data2['tumor-size'].replace(
        '40-44', '9').values[i]
    data2['tumor-size'].values[i] = data2['tumor-size'].replace(
        '45-49', '10').values[i]
    data2['tumor-size'].values[i] = data2['tumor-size'].replace(
        '50-54', '11').values[i]
    data2['tumor-size'].values[i] = data2['tumor-size'].replace(
        '55-59', '12').values[i]

    data2['inv-nodes'].values[i] = data2['inv-nodes'].replace(
        '0-2', '1').values[i]
    data2['inv-nodes'].values[i] = data2['inv-nodes'].replace(
        '3-5', '2').values[i]
    data2['inv-nodes'].values[i] = data2['inv-nodes'].replace(
        '6-8', '3').values[i]
    data2['inv-nodes'].values[i] = data2['inv-nodes'].replace(
        '9-11', '4').values[i]

    data2['inv-nodes'].values[i] = data2['inv-nodes'].replace(
        '12-14', '5').values[i]
    data2['inv-nodes'].values[i] = data2['inv-nodes'].replace(
        '15-17', '6').values[i]
    data2['inv-nodes'].values[i] = data2['inv-nodes'].replace(
        '18-20', '7').values[i]
    data2['inv-nodes'].values[i] = data2['inv-nodes'].replace(
        '21-23', '8').values[i]

    data2['inv-nodes'].values[i] = data2['inv-nodes'].replace(
        '24-26', '9').values[i]
    data2['inv-nodes'].values[i] = data2['inv-nodes'].replace(
        '27-29', '10').values[i]
    data2['inv-nodes'].values[i] = data2['inv-nodes'].replace(
        '30-32', '11').values[i]
    data2['inv-nodes'].values[i] = data2['inv-nodes'].replace(
        '33-35', '12').values[i]
    data2['inv-nodes'].values[i] = data2['inv-nodes'].replace(
        '36-39', '13').values[i]

    data2['node-caps'].values[i] = data2['node-caps'].replace(
        'yes', '1').values[i]
    data2['node-caps'].values[i] = data2['node-caps'].replace(
        'no', '2').values[i]

    data2['breast'].values[i] = data2['breast'].replace('right', '1').values[i]
    data2['breast'].values[i] = data2['breast'].replace('left', '2').values[i]

    data2['deg-malig'].values[i] = data2['deg-malig'].replace(3, '3').values[i]

    data2['breast-quad'].values[i] = data2['breast-quad'].replace(
        'left_up', '1').values[i]
    data2['breast-quad'].values[i] = data2['breast-quad'].replace(
        'right_up', '2').values[i]
    data2['breast-quad'].values[i] = data2['breast-quad'].replace(
        'left_low', '3').values[i]
    data2['breast-quad'].values[i] = data2['breast-quad'].replace(
        'right_low', '4').values[i]
    data2['breast-quad'].values[i] = data2['breast-quad'].replace(
        'central', '5').values[i]

    data2['irradiat'].values[i] = data2['irradiat'].replace(
        'yes', '1').values[i]
    data2['irradiat'].values[i] = data2['irradiat'].replace(
        'no', '2').values[i]

print("deg-malig = ", data2['deg-malig'].values)
# print(data2['tumor-size'].values)
# print(data2['inv-nodes'].values)

data2.to_csv(r'C:\Users\Lenovo\Desktop\Term 2-2565\Data Mining - 261448\Hw3 NN\data_preprocessing.txt',
             index=None, sep=',', mode='w')
