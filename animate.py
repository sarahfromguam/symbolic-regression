from symbolic_regression import SearchAlgorithms, VisualizeSearch, load_dataset


if __name__ == "__main__":

    dataset = load_dataset('data.txt')
    n_trials = 10000
    random_search = SearchAlgorithms()


    VisualizeSearch.plot_animation(dataset, 'figs/GA_animation_2.gif', random_search.run_ga_parallel, 
                                   [dataset, n_trials, 100, None])
    
    VisualizeSearch.plot_animation(dataset, 'figs/RMHC_100restarts_animation_2.gif', random_search.run_rmhc_parallel, 
                                   [dataset, n_trials, int(n_trials/10),None])
