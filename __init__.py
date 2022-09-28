try:
    import pyston_lite

    pyston_lite.enable()
    print('Enabled pyston-lite JIT')
except Exception as e:
    print(
        'Failed to init pyston-lite JIT, performance will be worse', type(e), e
    )
