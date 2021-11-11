from symbolic_regression import ExpressionHeap, SearchAlgorithms, VisualizeSearch, load_dataset
from matplotlib import pyplot as plt
from pathos.multiprocessing import ProcessPool

if __name__ == "__main__":

    dataset = [(x, 3*x) for x in range(30)]
    n_trials = 10000
    
    random_search=SearchAlgorithms(depth_dist=[3,5])


    # random_1 = random_search.get_random_heap()
    # print(1, '\t', random_1.heap)
    # random_2 = random_search.get_random_heap()
    # print(2, '\t', random_2.heap)

    # crossover = random_search.get_crossover(random_1, random_2)
    # print('Xed', '\t', crossover.heap)
    # print(crossover.evaluate(dataset))
    for i in range(1, 2):

        df, best_specimen = random_search.run_random_parallel(dataset, n_trials, 
                                                            num_nodes=None,
                                                            plot=True)
        results_subdir = 'results_simple/random_depth3to5'
        df.to_csv('{}/n{}_i{}.csv'.format(results_subdir, n_trials, i))
        expression_summary = '{}, MSE: {}'.format(best_specimen[-1].to_expr(),
                                                  df['best_scores'].to_list()[-1])
        with open('{}/n{}_i{}.txt'.format(results_subdir, n_trials, i), 'w') as f:
            f.write(expression_summary)
        print(expression_summary)
        plt.figure(figsize=(6, 6))
        VisualizeSearch.plot_f(best_specimen[-1], dataset)
        plt.savefig('{}/n{}_i{}.png'.format(results_subdir, n_trials, i), dpi=200)
        plt.show() 
    
        df, best_specimen = random_search.run_rmhc_parallel(dataset, n_trials, 
                                                            restart=int(n_trials/10), 
                                                            num_nodes=None,
                                                            plot=True)
        results_subdir = 'results_simple/rmhc_10restarts_depth3to5'
        df.to_csv('{}/n{}_i{}.csv'.format(results_subdir, n_trials, i))
        expression_summary = '{}, MSE: {}'.format(best_specimen[-1].to_expr(),
                                                  df['best_scores'].to_list()[-1])
        with open('{}/n{}_i{}.txt'.format(results_subdir, n_trials, i), 'w') as f:
            f.write(expression_summary)
        print(expression_summary)
        plt.figure(figsize=(6, 6))
        VisualizeSearch.plot_f(best_specimen[-1], dataset)
        plt.savefig('{}/n{}_i{}.png'.format(results_subdir, n_trials, i), dpi=200)
        plt.show() 
        
        df, best_specimen = random_search.run_ga_parallel(dataset, n_trials,  
                                                    num_nodes=None,
                                                            plot=True)
        results_subdir = 'results_simple/ga_depth3to5'
        df.to_csv('{}/n{}_i{}.csv'.format(results_subdir, n_trials, i))
        expression_summary = '{}, MSE: {}'.format(best_specimen[-1].to_expr(),
                                                  df['best_scores'].to_list()[-1])
        with open('{}/n{}_i{}.txt'.format(results_subdir, n_trials, i), 'w') as f:
            f.write(expression_summary)
        print(expression_summary)
        plt.figure(figsize=(6, 6))
        VisualizeSearch.plot_f(best_specimen[-1], dataset)
        plt.savefig('{}/n{}_i{}.png'.format(results_subdir, n_trials, i), dpi=200)
        plt.show() 


    # f = 'results/random/n10000_i4.csv'
    # df = pd.read_csv(f)
    # print(df['best_scores'].to_list()[-1])
    # expr = 'Mul(10, Pow(2, -1))'
    # expr = parse_expr(expr)
    # print(expr.evalf())
