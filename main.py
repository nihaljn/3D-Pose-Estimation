class Args:
    dataset = 'human36m'
    method = 'test_time_adapt' # or 'train' or 'viz'


def main():
    
    args = Args()
    if args.dataset == 'human36m':
        if args.method == 'test_time_adapt':
            from human36m.test_time_adapt import main as m
        elif args.method == 'train':
            from human36m.run import run as m
        elif args.method == 'viz':
            from human36m.viz_gen import main as m
            
    elif args.dataset == 'humaneva':
        if args.method == 'test_time_adapt':
            from humaneva.test_time_adapt import main as m
        elif args.method == 'train':
            from humaneva.run import run as m
        elif args.method == 'viz':
            from humaneva.viz_gen import main as m
    
    m()

if __name__ == '__main__':
    main()