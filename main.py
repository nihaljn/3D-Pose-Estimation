class Args:
    dataset = 'human36m'


def main():
    
#     args = Args()
#     if args.dataset == 'human36m':
#         from human36m.run import run
#     elif args.dataset == 'humaneva':
#         from humaneva.run import run
        
#     run()
    # from human36m.viz_gen import main as m
    # m()
    
    from human36m.test_time_adapt import main as m
    m()
    

if __name__ == '__main__':
    main()