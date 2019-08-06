import cProfile

def a():
    import Checkers.py
cProfile.run('a()',sort='cumtime')