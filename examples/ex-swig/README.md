Setup:
```
module load Anaconda/1.9.2-fasrc01 pcre/8.37-fasrc01 swig/3.0.10-fasrc01
python setup.py build_ext --inplace
```

The module should be accessible via python by using:
```
import simple
simple.create_list(10)
```
