#!/usr/bin/env python

execfile('test_header.py')

# G: type
TestCase(
    'Test if G of valid type succeeds',
    should_succeed=True,
    G=100).run()

TestCase(
    'Test if G of invalid type fails',
    should_succeed=False,
    G=100.0).run()

TestCase(
    'Test if G of invalid type fails',
    should_succeed=False,
    G=True).run()

TestCase(
    'Test if G of invalid type fails',
    should_succeed=False,
    G=None).run()

TestCase(
    'Test if G of invalid type fails',
    should_succeed=False,
    G='100').run()

#TestCase(
#    'Test if G of invalid type fails',
#    should_succeed=False,
#    G=(100)).run() # ---> Interpreted as 100

TestCase(
    'Test if G of invalid type fails',
    should_succeed=False,
    G=[100]).run()

TestCase(
    'Test if G of invalid type fails',
    should_succeed=False,
    G={100}).run()

# G: value
TestCase(
    'Test if G > 50 succeeds',
    should_succeed=True,
    G=51).run()

TestCase(
    'Test if G = 50 succeeds',
    should_succeed=True,
    G=50).run()

TestCase(
    'Test if G < 50 fails',
    should_succeed=False,
    G=49).run()

TestCase(
    'Test if G = 0 fails',
    should_succeed=False,
    G=0).run()

TestCase(
    'Test if G < 0 fails',
    should_succeed=False,
    G=-50).run()

# alpha: type
TestCase(
    'Test if alpha of valid type succeeds',
    should_succeed=True,
    alpha=3).run()

TestCase(
    'Test if alpha of invalid type fails',
    should_succeed=False,
    alpha=3.0).run()

TestCase(
    'Test if alpha of invalid type fails',
    should_succeed=False,
    alpha=True).run()

TestCase(
    'Test if alpha of invalid type fails',
    should_succeed=False,
    alpha=None).run()

TestCase(
    'Test if alpha of invalid type fails',
    should_succeed=False,
    alpha='3').run()

#TestCase(
#    'Test if alpha of invalid type fails',
#    should_succeed=False,
#    alpha=(3)).run() # ---> Interpreted as 3

TestCase(
    'Test if alpha of invalid type fails',
    should_succeed=False,
    alpha=[3]).run()

TestCase(
    'Test if alpha of invalid type fails',
    should_succeed=False,
    alpha={3}).run()

# alpha: value
TestCase(
    'Test if alpha > 5 fails',
    should_succeed=False,
    alpha=6).run()

TestCase(
    'Test if alpha = 5 succeeds',
    should_succeed=True,
    alpha=5).run()

TestCase(
    'Test if alpha < 5 succeeds',
    should_succeed=True,
    alpha=4).run()

TestCase(
    'Test if alpha > 1 succeeds',
    should_succeed=True,
    alpha=2).run()

TestCase(
    'Test if alpha = 1 succeeds',
    should_succeed=True,
    alpha=1).run()

TestCase(
    'Test if alpha < 1 fails',
    should_succeed=False,
    alpha=0).run()

TestCase(
    'Test if alpha < 0 succeeds',
    should_succeed=False,
    alpha=-1).run()

# bbox: type
TestCase(
    'Test if bbox of valid type succeeds',
    should_succeed=True,
    bbox=[-6,6]).run()

TestCase(
    'Test if bbox of invalid type fails',
    should_succeed=False,
    bbox=(-6,6)).run()

TestCase(
    'Test if bbox of invalid type fails',
    should_succeed=False,
    bbox={-6,6}).run()

TestCase(
    'Test if bbox of invalid type fails',
    should_succeed=False,
    bbox=-6).run()

TestCase(
    'Test if bbox of invalid type fails',
    should_succeed=False,
    bbox=-6.0).run()

TestCase(
    'Test if bbox of invalid type fails',
    should_succeed=False,
    bbox=True).run()

TestCase(
    'Test if bbox of invalid type fails',
    should_succeed=False,
    bbox=None).run()

TestCase(
    'Test if bbox of invalid type fails',
    should_succeed=False,
    bbox='-6,6').run()

# bbox: length
TestCase(
    'Test if bbox of length = 2 succeeds',
    should_succeed=True,
    bbox=[-6,6]).run()

TestCase(
    'Test if bbox of length > 2 fails',
    should_succeed=False,
    bbox=[-6,6,1]).run()

TestCase(
    'Test if bbox of length < 2 fails',
    should_succeed=False,
    bbox=[-6]).run()

#TestCase(
#    'Test if bbox with missing boundary fails', 
#    should_succeed=False,
#    bbox=[-6, ]).run() # --> Interpreted as [-6]

#TestCase( 
#    'Test if bbox with missing boundary fails', 
#    should_succeed=False,
#    bbox=[ ,6]).run() # --> SyntaxError

# bbox: value
TestCase(
    'Test if bbox containing numbers succeeds', 
    should_succeed=True,
    bbox=[-6,6]).run()

TestCase(
    'Test if bbox containing things other than numbers fails', 
    should_succeed=False,
    bbox=[True,6]).run()

TestCase(
    'Test if bbox containing things other than numbers fails', 
    should_succeed=False,
    bbox=[-6,True]).run()

TestCase(
    'Test if bbox containing things other than numbers fails', 
    should_succeed=False,
    bbox=[None,6]).run()

TestCase(
    'Test if bbox containing things other than numbers fails', 
    should_succeed=False,
    bbox=[-6,None]).run()

TestCase(
    'Test if bbox containing things other than numbers fails', 
    should_succeed=False,
    bbox=['-6',6]).run()

TestCase(
    'Test if bbox containing things other than numbers fails', 
    should_succeed=False,
    bbox=[-6,'6']).run()

# bbox: ordering
TestCase(
    'Test if bbox with ordered boundaries succeeds', 
    should_succeed=True,
    bbox=[-6,6]).run()

TestCase(
    'Test if bbox with misordered boundaries fails', 
    should_succeed=False,
    bbox=[6,-6]).run()

# periodic: type
TestCase(
    'Test if periodic of valid type succeeds', 
    should_succeed=True,
    periodic=False).run()

TestCase(
    'Test if periodic of invalid type fails', 
    should_succeed=False,
    periodic=None).run()

TestCase(
    'Test if periodic of invalid type fails', 
    should_succeed=False,
    periodic='False').run()

TestCase(
    'Test if periodic of invalid type fails', 
    should_succeed=False,
    periodic=0).run()

TestCase(
    'Test if periodic of invalid type fails', 
    should_succeed=False,
    periodic=0.0).run()

#TestCase(
#    'Test if periodic of invalid type fails', 
#    should_succeed=False,
#    periodic=(0)).run() # --- > Inperpreted as 0

TestCase(
    'Test if periodic of invalid type fails', 
    should_succeed=False,
    periodic=[0]).run()

TestCase(
    'Test if periodic of invalid type fails', 
    should_succeed=False,
    periodic={0}).run()

# periodic: value
TestCase(
    'Test if periodic = True succeeds', 
    should_succeed=True,
    periodic=True).run()

TestCase(
    'Test if periodic = False succeeds', 
    should_succeed=True,
    periodic=False).run()

# Laplace: type
TestCase(
    'Test if Laplace of valid type succeeds', 
    should_succeed=True,
    Laplace=True).run()

TestCase(
    'Test if Laplace of invalid type fails', 
    should_succeed=False,
    Laplace=None).run()

TestCase(
    'Test if Laplace of invalid type fails', 
    should_succeed=False,
    Laplace='True').run()

TestCase(
    'Test if Laplace of invalid type fails', 
    should_succeed=False,
    Laplace=1).run()

TestCase(
    'Test if periodic of invalid type fails', 
    should_succeed=False,
    Laplace=1.0).run()

#TestCase(
#    'Test if periodic of invalid type fails', 
#    should_succeed=False,
#    Laplace=(1)).run() # ---> Interpreted as 1

TestCase(
    'Test if periodic of invalid type fails', 
    should_succeed=False,
    Laplace=[1]).run()

TestCase(
    'Test if periodic of invalid type fails', 
    should_succeed=False,
    Laplace={1}).run()

# Laplace: value
TestCase(
    'Test if Laplace = True succeeds', 
    should_succeed=True,
    Laplace=True).run()

TestCase(
    'Test if Laplace = False succeeds', 
    should_succeed=True,
    Laplace=False).run()

# print_t: type
TestCase(
    'Test if print_t of valid type succeeds', 
    should_succeed=True,
    print_t=False).run()

TestCase(
    'Test if print_t of invalid type fails', 
    should_succeed=False,
    print_t=None).run()

TestCase(
    'Test if print_t of invalid type fails', 
    should_succeed=False,
    print_t='False').run()

TestCase(
    'Test if print_t of invalid type fails', 
    should_succeed=False,
    print_t=0).run()

TestCase(
    'Test if print_t of invalid type fails', 
    should_succeed=False,
    print_t=0.0).run()

#TestCase(
#    'Test if print_t of invalid type fails', 
#    should_succeed=False,
#    print_t=(0)).run() # ---> Interpreted as 0

TestCase(
    'Test if print_t of invalid type fails', 
    should_succeed=False,
    print_t=[0]).run()

TestCase(
    'Test if print_t of invalid type fails', 
    should_succeed=False,
    print_t={0}).run()

# print_t: value
TestCase(
    'Test if print_t = True succeeds', 
    should_succeed=True,
    print_t=True).run()

TestCase(
    'Test if print_t = False succeeds', 
    should_succeed=True,
    print_t=False).run()

# tollerance: type
TestCase(
    'Test if tollerance of valid type succeeds',
    should_succeed=True,
    tollerance=1E-3).run()

TestCase(
    'Test if tollerance of invalid type fails',
    should_succeed=False,
    tollerance=1).run()

TestCase(
    'Test if tollerance of invalid type fails',
    should_succeed=False,
    tollerance=True).run()

TestCase(
    'Test if tollerance of invalid type fails',
    should_succeed=False,
    tollerance=None).run()

TestCase(
    'Test if tollerance of invalid type fails',
    should_succeed=False,
    tollerance='1E-3').run()

#TestCase(
#    'Test if tollerance of invalid type fails',
#    should_succeed=False,
#    tollerance=(1E-3)).run() # ---> Interpreted as 1E-3

TestCase(
    'Test if tollerance of invalid type fails',
    should_succeed=False,
    tollerance=[1E-3]).run()

TestCase(
    'Test if tollerance of invalid type fails',
    should_succeed=False,
    tollerance={1E-3}).run()

# tollerance: value
TestCase(
    'Test if tollerance > 0 succeeds',
    should_succeed=True,
    tollerance=1E-3).run()

TestCase(
    'Test if tollerance = 0 fails',
    should_succeed=False,
    tollerance=0.0).run()

TestCase(
    'Test if tollerance < 0 fails',
    should_succeed=False,
    tollerance=-1E-3).run()

# resolution: type
TestCase(
    'Test if resolution of valid type succeeds',
    should_succeed=True,
    resolution=1E-1).run()

TestCase(
    'Test if resolutione of invalid type fails',
    should_succeed=False,
    resolution=1).run()

TestCase(
    'Test if resolutione of invalid type fails',
    should_succeed=False,
    resolution=True).run()

TestCase(
    'Test if resolutione of invalid type fails',
    should_succeed=False,
    resolution=None).run()

TestCase(
    'Test if resolution of invalid type fails',
    should_succeed=False,
    resolution='1E-1').run()

#TestCase(
#    'Test if resolution of invalid type fails',
#    should_succeed=False,
#    resolution=(1E-1)).run() # ---> Interpreted as 1E-1

TestCase(
    'Test if resolution of invalid type fails',
    should_succeed=False,
    resolution=[1E-1]).run()

TestCase(
    'Test if resolution of invalid type fails',
    should_succeed=False,
    resolution={1E-1}).run()

# resolution: value
TestCase(
    'Test if resolution > 0 succeeds',
    should_succeed=True,
    resolution=1E-1).run()

TestCase(
    'Test if resolution = 0 fails',
    should_succeed=False,
    resolution=0.0).run()

TestCase(
    'Test if resolution < 0 fails',
    should_succeed=False,
    resolution=-1E-1).run()

# num_samples: type
TestCase(
    'Test if num_samples of valid type succeeds',
    should_succeed=True,
    num_samples=0).run()

TestCase(
    'Test if num_samples of invalid type fails',
    should_succeed=False,
    num_samples=0.0).run()

TestCase(
    'Test if num_samples of invalid type fails',
    should_succeed=False,
    num_samples=True).run()

TestCase(
    'Test if num_samples of invalid type fails',
    should_succeed=False,
    num_samples=None).run()

TestCase(
    'Test if num_samples of invalid type fails',
    should_succeed=False,
    num_samples='0').run()

#TestCase(
#    'Test if num_samples of invalid type fails',
#    should_succeed=False,
#    num_samples=(0)).run() # ---> Interpreted as 0

TestCase(
    'Test if num_samples of invalid type fails',
    should_succeed=False,
    num_samples=[0]).run()

TestCase(
    'Test if num_samples of invalid type fails',
    should_succeed=False,
    num_samples={0}).run()

# num_samples: value
TestCase(
    'Test if num_samples > 0 succeeds',
    should_succeed=True,
    num_samples=1).run()

TestCase(
    'Test if num_samples = 0 succeeds',
    should_succeed=True,
    num_samples=0).run()

TestCase(
    'Test if num_samples < 0 fails',
    should_succeed=False,
    num_samples=-1).run()

# fix_t_at_t_star: type
TestCase(
    'Test if fix_t_at_t_star of valid type succeeds', 
    should_succeed=True,
    fix_t_at_t_star=False).run()

TestCase(
    'Test if fix_t_at_t_star of invalid type fails', 
    should_succeed=False,
    fix_t_at_t_star=None).run()

TestCase(
    'Test if fix_t_at_t_star of invalid type fails', 
    should_succeed=False,
    fix_t_at_t_star='False').run()

TestCase(
    'Test if fix_t_at_t_star of invalid type fails', 
    should_succeed=False,
    fix_t_at_t_star=0).run()

TestCase(
    'Test if fix_t_at_t_star of invalid type fails', 
    should_succeed=False,
    fix_t_at_t_star=0.0).run()

#TestCase(
#    'Test if fix_t_at_t_star of invalid type fails', 
#    should_succeed=False,
#    fix_t_at_t_star=(0)).run() # ---> Interpreted as 0

TestCase(
    'Test if fix_t_at_t_star of invalid type fails', 
    should_succeed=False,
    fix_t_at_t_star=[0]).run()

TestCase(
    'Test if fix_t_at_t_star of invalid type fails', 
    should_succeed=False,
    fix_t_at_t_star={0}).run()

# fix_t_at_t_star: value
TestCase(
    'Test if fix_t_at_t_star = True succeeds', 
    should_succeed=True,
    fix_t_at_t_star=True).run()

TestCase(
    'Test if fix_t_at_t_star = False succeeds', 
    should_succeed=True,
    fix_t_at_t_star=False).run()

# deft_seed: type
TestCase(
    'Test if deft_seed of valid type succeeds',
    should_succeed=True,
    deft_seed=None).run()

TestCase(
    'Test if deft_seed of valid type succeeds',
    should_succeed=True,
    deft_seed=0).run()

TestCase(
    'Test if deft_seed of invalid type fails',
    should_succeed=False,
    deft_seed=0.0).run()

TestCase(
    'Test if deft_seed of invalid type fails',
    should_succeed=False,
    deft_seed=True).run()

TestCase(
    'Test if deft_seed of invalid type fails',
    should_succeed=False,
    deft_seed='0.0').run()

#TestCase(
#    'Test if deft_seed of invalid type fails',
#    should_succeed=False,
#    deft_seed=(0)).run() # ---> Interpreted as 0

TestCase(
    'Test if deft_seed of invalid type fails',
    should_succeed=False,
    deft_seed=[0]).run()

TestCase(
    'Test if deft_seed of invalid type fails',
    should_succeed=False,
    deft_seed={0}).run()

# deft_seed: value
TestCase(
    'Test if deft_seed > 2**32-1 fails',
    should_succeed=False,
    deft_seed=2**32).run()

TestCase(
    'Test if deft_seed = 2**32-1 succeeds',
    should_succeed=True,
    deft_seed=2**32-1).run()

TestCase(
    'Test if deft_seed < 2**32-1 succeeds',
    should_succeed=True,
    deft_seed=2**32-2).run()

TestCase(
    'Test if deft_seed > 0 succeeds',
    should_succeed=True,
    deft_seed=1).run()

TestCase(
    'Test if deft_seed = 0 succeeds',
    should_succeed=True,
    deft_seed=0).run()

TestCase(
    'Test if deft_seed < 0 fails',
    should_succeed=False,
    deft_seed=-1).run()
