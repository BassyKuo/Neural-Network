# Nolearn MNIST example
You can run the tiny example `simple_mnist.py` if you are first time to contact it:

```
$ python simple_mnist.py
```

The result will be like this:
```
# Neural Network with 122154 learnable parameters

## Layer information

name        size        total    cap.Y    cap.X    cov.Y    cov.X
----------  --------  -------  -------  -------  -------  -------
input0      1x28x28       784   100.00   100.00   100.00   100.00
conv2d1     32x26x26    21632   100.00   100.00    10.71    10.71
maxpool2d2  32x13x13     5408   100.00   100.00    10.71    10.71
conv2d3     64x11x11     7744    85.71    85.71    25.00    25.00
conv2d4     64x9x9       5184    54.55    54.55    39.29    39.29
maxpool2d5  64x4x4       1024    54.55    54.55    39.29    39.29
conv2d6     96x2x2        384    63.16    63.16    67.86    67.86
maxpool2d7  96x1x1         96    63.16    63.16    67.86    67.86
dense8      64             64   100.00   100.00   100.00   100.00
dropout9    64             64   100.00   100.00   100.00   100.00
dense10     64             64   100.00   100.00   100.00   100.00
dense11     10             10   100.00   100.00   100.00   100.00

Explanation
    X, Y:    image dimensions
    cap.:    learning capacity
    cov.:    coverage of image
    magenta: capacity too low (<1/6)
    cyan:    image coverage too high (>100%)
    red:     capacity too low and coverage too high


  epoch    trn loss    val loss    trn/val    valid acc  dur
-------  ----------  ----------  ---------  -----------  -----
      1     0.68008     0.11388    5.97186      0.96635  6.25s
      2     0.16667     0.08609    1.93598      0.97593  5.97s
      3     0.11589     0.07747    1.49599      0.97984  6.28s
      4     0.08812     0.07019    1.25546      0.98234  6.33s
      5     0.07151     0.06582    1.08641      0.98267  6.25s
      6     0.06319     0.06136    1.02982      0.98601  6.25s
      7     0.05554     0.05917    0.93871      0.98567  5.84s
      8     0.04899     0.06015    0.81444      0.98651  6.28s
      9     0.04474     0.05213    0.85813      0.98867  6.29s
     10     0.03853     0.05597    0.68836      0.98709  6.26s
Start to test.....
The accuracy of this network is: 0.99
```
