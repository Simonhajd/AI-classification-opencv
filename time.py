from timeit import Timer

# first argument is the code to be run, the second "setup" argument is only run once,
# and it not included in the execution time.
t = Timer("""x.index(123)""", setup="""x = range(1000)""")

print(t.timeit()) # prints float, for example 5.8254
# ..or..
print(t.timeit(1000)) # repeat 1000 times instead of the default 1million