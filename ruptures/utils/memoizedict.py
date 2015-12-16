class MemoizeDict(dict):
    """Be careful: a function which has been decorated with MemoizeDict cannot
    accept names in its arguments:
     foo(a=1, b='abc') ---> won't work
     foo(1, 'abc') ---> ok
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        result = self[key] = self.func(*key)
        return result

#
# Sample use
#
# if __name__ == "__main__":
#     @MemoizeDict
#     def foo(a, b):
#         return a * b
#
#     print(foo(2, 4))
#     print(foo('hi', 3))
#     print(foo)
