# Machine-Learning-Deep-Learning-Logistic-Regession

b0 is: [-4.50163542]

b1 is: [[1.00401882]]


Logistics regress credit data analysis

creditData.head--------------------------------

   clientid        income        age         loan       LTI  default
0         1  66155.925095  59.017015  8106.532131  0.122537        0
1         2  34415.153966  48.117153  6564.745018  0.190752        0
2         3  57317.170063  63.108049  8020.953296  0.139940        0
3         4  42709.534201  45.751972  6103.642260  0.142911        0
4         5  66952.688845  18.584336  8770.099235  0.130989        1

creditData.describe--------------------------------

          clientid        income  ...          LTI      default
count  2000.000000   2000.000000  ...  2000.000000  2000.000000
mean   1000.500000  45331.600018  ...     0.098403     0.141500
std     577.494589  14326.327119  ...     0.057620     0.348624
min       1.000000  20014.489470  ...     0.000049     0.000000
25%     500.750000  32796.459717  ...     0.047903     0.000000
50%    1000.500000  45789.117313  ...     0.099437     0.000000
75%    1500.250000  57791.281668  ...     0.147585     0.000000
max    2000.000000  69995.685578  ...     0.199938     1.000000

[8 rows x 6 columns]

creditData.corr--------------------------------

          clientid    income       age      loan       LTI   default
clientid  1.000000  0.039280 -0.030341  0.018931  0.002538 -0.020145
income    0.039280  1.000000 -0.034984  0.441117 -0.019862  0.002284
age      -0.030341 -0.034984  1.000000  0.006561  0.021588 -0.444765
loan      0.018931  0.441117  0.006561  1.000000  0.847495  0.377160
LTI       0.002538 -0.019862  0.021588  0.847495  1.000000  0.433261
default  -0.020145  0.002284 -0.444765  0.377160  0.433261  1.000000

confusion matrix
[[521   7]
 [ 30  42]]

accuracy score
0.9383333333333334
