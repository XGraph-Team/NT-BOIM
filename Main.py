from methods import *
warnings.filterwarnings("ignore")
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

# graph_size = 1000
# candidate_size = 50 # candidate pool size
# number_of_sources = 3 # budget for IM
# num_iterations = 300 # budget for BO
# actual_time_step_size = 5 # diffusion parameter
# allowed_shortest_distance = 1 # shortest distance between sources for filtering
# num_of_sims = 10 # number of rounds for MC simulation
# number_of_clusters = 20

diffusion_model = "ic" # "ic" or "lt"
num_of_sims = 100
num_iterations = 300
num_of_repeats = 5
number_of_sources = 3
print("diffusion_model", diffusion_model)
print("number of sims", num_of_sims)
print("number of iterations", num_iterations)


################################################
# Global parameters
################################################

graphs_info = {
    # 'synn_small': synn_small, # max degree == 1
    # 'synn': synn, # max degree == 1
    # 'SW_400': lambda: connSW(400), # max degree == 1
    # 'SW_800': lambda: connSW(800), # max degree == 2
    # 'SW_1200': lambda: connSW(1200), # max degree == 3 or more ?
    # 'SW_2400': lambda: connSW(2400), # max degree == 3 or more ?
    # 'SW_4800': lambda: connSW(4800), # max degree == 3 or more ?
    # 'ER_3000': lambda: ER(3000), # max degree == 1
    # 'BA_11': lambda: BA(11), # max degree == 1
    # 'BA_12': lambda: BA(12), # max degree == 2
    # 'BA_13': lambda: BA(13), # max degree == 3 or more ?
    # 'BA_14': lambda: BA(14), # max degree == 3 or more ?; 8 combinations X Y Z
    # 'BA_3000': lambda: BA(3000), # max degree == 3 or more ?
    'CiteSeer': CiteSeer, # max degree == 3 or more ?
    'Cora': Cora, # max degree == 3 or more ?
    # 'PubMed': PubMed # max degree == 3 or more ?
    # 'power_law_1000_3': lambda: power_law(1000,3), 
}

methods = {
    ############################################
    # NOT READY
    # 'BOIM_fourier_hodgelaplacians_3_simplcies_3_maxdim_3_degree': {'func': BOIM_fourier_hodgelaplacians_3_simplcies_3_maxdim_3_degree, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_1_nl_sparse': {'func': BOIM_fourier_hodgelaplacians_1_nl_sparse, 'needs_simplices': True, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_0_1_combined_padding_nl_sparse': {'func': BOIM_fourier_hodgelaplacians_0_1_combined_padding_nl_sparse, 'needs_simplices': True, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_0_1_combined_heat_kernel_nl_sparse': {'func': BOIM_fourier_hodgelaplacians_0_1_combined_heat_kernel_nl_sparse, 'needs_simplices': True, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_0_1_combined_transform_features': {'func': BOIM_fourier_hodgelaplacians_0_1_combined_transform_features, 'needs_simplices': True, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_0_1_combined_mapping': {'func': BOIM_fourier_hodgelaplacians_0_1_combined_mapping, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_2': {'func': BOIM_fourier_hodgelaplacians_2, 'needs_simplices': True, 'is_boim': True},
    # 'BOIM_fourier_graph_to_hodge_3': {'func': BOIM_fourier_graph_to_hodge_3, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_x2_y2_z1': {'func': BOIM_vanilla_hybrid_x2_y2_z1, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_x2_y2_z2': {'func': BOIM_vanilla_hybrid_x2_y2_z2, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_x3_y3_z2': {'func': BOIM_vanilla_hybrid_x3_y3_z2, 'needs_simplices': False, 'is_boim': True}        'BOIM_vanilla_hybrid_g_to_h_2': {'func': BOIM_vanilla_hybrid_g_to_h_2, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_g_to_h_3': {'func': BOIM_vanilla_hybrid_g_to_h_3, 'needs_simplices': False, 'is_boim': True},
    ############################################
    # 'BOIM_fourier_hodgelaplacians_1_simplcies_1_maxdim_0_degree': {'func': BOIM_fourier_hodgelaplacians_1_simplcies_1_maxdim_0_degree, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_1_simplcies_1_maxdim_1_degree': {'func': BOIM_fourier_hodgelaplacians_1_simplcies_1_maxdim_1_degree, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_2_simplcies_2_maxdim_1_degree': {'func': BOIM_fourier_hodgelaplacians_2_simplcies_2_maxdim_1_degree, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_2_simplcies_2_maxdim_2_degree': {'func': BOIM_fourier_hodgelaplacians_2_simplcies_2_maxdim_2_degree, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_3_simplcies_3_maxdim_2_degree': {'func': BOIM_fourier_hodgelaplacians_3_simplcies_3_maxdim_2_degree, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_0_max_dim_1_nl_sparse': {'func': BOIM_fourier_hodgelaplacians_0_max_dim_1_nl_sparse, 'needs_simplices': True, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_0_max_dim_2_nl_sparse': {'func': BOIM_fourier_hodgelaplacians_0_max_dim_2_nl_sparse, 'needs_simplices': True, 'is_boim': True},
    # 'BOIM_fourier_graph_to_hodge_0': {'func': BOIM_fourier_graph_to_hodge_0, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_1_simplcies_1_maxdim_0_degree': {'func': BOIM_fourier_hodgelaplacians_1_simplcies_1_maxdim_0_degree, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_1_simplcies_1_maxdim_1_degree': {'func': BOIM_fourier_hodgelaplacians_1_simplcies_1_maxdim_1_degree, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_2_simplcies_2_maxdim_1_degree': {'func': BOIM_fourier_hodgelaplacians_2_simplcies_2_maxdim_1_degree, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_2_simplcies_2_maxdim_2_degree': {'func': BOIM_fourier_hodgelaplacians_2_simplcies_2_maxdim_2_degree, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_3_simplcies_3_maxdim_2_degree': {'func': BOIM_fourier_hodgelaplacians_3_simplcies_3_maxdim_2_degree, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_0_max_dim_1_nl_sparse': {'func': BOIM_fourier_hodgelaplacians_0_max_dim_1_nl_sparse, 'needs_simplices': True, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_0_max_dim_2_nl_sparse': {'func': BOIM_fourier_hodgelaplacians_0_max_dim_2_nl_sparse, 'needs_simplices': True, 'is_boim': True},
    # 'BOIM_fourier_graph_to_hodge_0': {'func': BOIM_fourier_graph_to_hodge_0, 'needs_simplices': False, 'is_boim': True},
    # # 'BOIM_fourier_graph_to_hodge_1': {'func': BOIM_fourier_graph_to_hodge_1, 'needs_simplices': False, 'is_boim': True},
    # # 'BOIM_fourier_graph_to_hodge_2': {'func': BOIM_fourier_graph_to_hodge_2, 'needs_simplices': False, 'is_boim': True},
    ############################################
    # 'BOIM_fourier_hodgelaplacians_1_simplcies_1_maxdim_1_degree_mapping': {'func': BOIM_fourier_hodgelaplacians_1_simplcies_1_maxdim_1_degree_mapping, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_0_1_combined_padding': {'func': BOIM_fourier_hodgelaplacians_0_1_combined_padding, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_0_1_combined_heat_kernel': {'func': BOIM_fourier_hodgelaplacians_0_1_combined_heat_kernel, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_combined_0_1_max_dim_1_N_N': {'func': BOIM_fourier_hodgelaplacians_combined_0_1_max_dim_1_N_N, 'needs_simplices': True, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_combined_0_1_max_dim_2_N_N': {'func': BOIM_fourier_hodgelaplacians_combined_0_1_max_dim_2_N_N, 'needs_simplices': True, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_combined_0_1_max_dim_1_M_M': {'func': BOIM_fourier_hodgelaplacians_combined_0_1_max_dim_1_M_M, 'needs_simplices': True, 'is_boim': True},
    # 'BOIM_fourier_hodgelaplacians_combined_0_1_max_dim_2_M_M': {'func': BOIM_fourier_hodgelaplacians_combined_0_1_max_dim_2_M_M, 'needs_simplices': True, 'is_boim': True},    
    ############################################
    # 'BOIM_fourier': {'func': BOIM_fourier, 'needs_simplices': False, 'is_boim': True},
    'BOIM_fourier_ntk': {'func': BOIM_fourier_ntk, 'needs_simplices': False, 'is_boim': True},
    # # # 'BOIM_fourier_cuda': {'func': BOIM_fourier_cuda, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_non_normalized': {'func': BOIM_fourier_non_normalized, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_no_GSS': {'func': BOIM_fourier_no_GSS, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_no_GSS_random_ntk': {'func': BOIM_fourier_no_GSS_random_ntk, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_no_GSS_normal_ntk': {'func': BOIM_fourier_no_GSS_normal_ntk, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_no_GSS_cluster_ntk': {'func': BOIM_fourier_no_GSS_cluster_ntk, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_no_GSS_stratified_ntk': {'func': BOIM_fourier_no_GSS_stratified_ntk, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_no_filtering': {'func': BOIM_fourier_no_filtering, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_no_GSS_top': {'func': BOIM_fourier_no_GSS_top, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_fourier_no_GSS_top_centrality_normalized': {'func': BOIM_fourier_no_GSS_top_centrality_normalized, 'needs_simplices': False, 'is_boim': True},
    # ############################################s
    # # # 'BOIM_vanilla_hybrid_tune_centrality_process': {'func': BOIM_vanilla_hybrid_tune_centrality_process, 'needs_simplices': False, 'is_boim': True},
    # # # 'BOIM_vanilla_hybrid_tune_centrality_threads': {'func': BOIM_vanilla_hybrid_tune_centrality_threads, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_tune_centrality': {'func': BOIM_vanilla_hybrid_tune_centrality, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_adaptive': {'func': BOIM_vanilla_hybrid_adaptive, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_adaptive_centrality_normalized': {'func': BOIM_vanilla_hybrid_adaptive_centrality_normalized, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_adaptive_centrality_normalized_with_early_stopping': {'func': BOIM_vanilla_hybrid_adaptive_centrality_normalized_with_early_stopping, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_random_centrality_normalized': {'func': BOIM_vanilla_random_centrality_normalized, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_tune_centrality_normalized': {'func': BOIM_vanilla_hybrid_tune_centrality_normalized, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_tune_degree_betweenness_closeness_eigenvector': {'func': BOIM_vanilla_hybrid_tune_degree_betweenness_closeness_eigenvector, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_random_centrality': {'func': BOIM_vanilla_random_centrality, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_random': {'func': BOIM_vanilla_random, 'needs_simplices': False, 'is_boim': True},
    # #'BOIM_vanilla_partial_random': {'func': BOIM_vanilla_partial_random, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_random_L': {'func': BOIM_vanilla_random_L, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_normal_centrality_normalized': {'func': BOIM_vanilla_normal_centrality_normalized, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_top_nngp': {'func': BOIM_vanilla_top_nngp, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_random_nngp': {'func': BOIM_vanilla_random_nngp, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_normal_nngp': {'func': BOIM_vanilla_normal_nngp, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_stratified_nngp': {'func': BOIM_vanilla_stratified_nngp, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_cluster_nngp': {'func': BOIM_vanilla_cluster_nngp, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_random_ntk': {'func': BOIM_vanilla_random_ntk, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_normal_ntk': {'func': BOIM_vanilla_normal_ntk, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_stratified_ntk': {'func': BOIM_vanilla_stratified_ntk, 'needs_simplices': False, 'is_boim': True},    
    # 'BOIM_vanilla_cluster_ntk': {'func': BOIM_vanilla_cluster_ntk, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_top_ntk': {'func': BOIM_vanilla_top_ntk, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_top_no_gp': {'func': BOIM_vanilla_top_no_gp, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_top_centrality_normalized': {'func': BOIM_vanilla_top_centrality_normalized, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_stratified_centrality_normalized': {'func': BOIM_vanilla_stratified_centrality_normalized, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_cluster_centrality_normalized': {'func': BOIM_vanilla_cluster_centrality_normalized, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_stratified': {'func': BOIM_vanilla_stratified, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_cluster': {'func': BOIM_vanilla_cluster, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_normal': {'func': BOIM_vanilla_normal, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_top': {'func': BOIM_vanilla_top, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid': {'func': BOIM_vanilla_hybrid, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_tune': {'func': BOIM_vanilla_hybrid_tune, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_L': {'func': BOIM_vanilla_hybrid_L, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_tune_L': {'func': BOIM_vanilla_hybrid_tune_L, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_nnl': {'func': BOIM_vanilla_hybrid_nnl, 'needs_simplices': False, 'is_boim': True},
    # ############################################
    # 'BOIM_vanilla_hybrid_x1_y1_z0': {'func': BOIM_vanilla_hybrid_x1_y1_z0, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_x1_y1_z1': {'func': BOIM_vanilla_hybrid_x1_y1_z1, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_g_to_h_0': {'func': BOIM_vanilla_hybrid_g_to_h_0, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_g_to_h_1': {'func': BOIM_vanilla_hybrid_g_to_h_1, 'needs_simplices': False, 'is_boim': True},
    # ############################################
    # 'BOIM_vanilla_hybrid_tune_centrality_dynamic': {'func': BOIM_vanilla_hybrid_tune_centrality_dynamic, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_tune_centrality_multiGP': {'func': BOIM_vanilla_hybrid_tune_centrality_multiGP, 'needs_simplices': False, 'is_boim': True},
    # 'BOIM_vanilla_hybrid_tune_centrality_dynamic_multiGP': {'func': BOIM_vanilla_hybrid_tune_centrality_dynamic_multiGP, 'needs_simplices': False, 'is_boim': True},
    ############################################
    # 'greedy': {'func': greedy, 'needs_simplices': False, 'is_boim': False},
    # 'celf': {'func': celf, 'needs_simplices': False, 'is_boim': False},
    # 'celfpp': {'func': celfpp, 'needs_simplices': False, 'is_boim': False},
    # 'IMM': {'func': IMM, 'needs_simplices': False, 'is_boim': False},
    # 'IMRank': {'func': IMRank, 'needs_simplices': False, 'is_boim': False},
    # 'RIS': {'func': RIS, 'needs_simplices': False, 'is_boim': False},
    # ############################################
    # 'degreeDis': {'func': degreeDis, 'needs_simplices': False, 'is_boim': False},
    # 'degree': {'func': degree, 'needs_simplices': False, 'is_boim': False},
    # 'eigen': {'func': eigen, 'needs_simplices': False, 'is_boim': False},
    # 'pi': {'func': pi, 'needs_simplices': False, 'is_boim': False},
    # 'sigma': {'func': sigma, 'needs_simplices': False, 'is_boim': False},
    # 'SoboldegreeDis': {'func': SoboldegreeDis, 'needs_simplices': False, 'is_boim': False},
    # 'Soboldeg': {'func': Soboldeg, 'needs_simplices': False, 'is_boim': False},
    # 'Netshield': {'func': Netshield, 'needs_simplices': False, 'is_boim': False},
}

for graph_name, graph_func in graphs_info.items():
    print('============================================')
    print(graph_name)
    G, config = graph_func()
    print("G",G)
    get_graph_infomation(G)
    simplices = generate_simplices_tetrahedron(G)

    results = {method_name: [] for method_name in methods.keys()}
    runtime_results = {method_name: [] for method_name in methods.keys()}

    for i in range(num_of_repeats):
        print('Repeating experiment: ', i)
        for method_name, method_details in methods.items():
            print(f'Running {method_name}...')  # This line will show which method is currently being processed.
            start_time = time.time()
            if method_details['is_boim']:
                if method_details['needs_simplices']:
                    result = method_details['func'](G, simplices, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters)
                elif 'BOIM_fourier_no_GSS' in method_name:
                # elif 'BOIM_fourier_no_GSS' == method_name or 'BOIM_fourier_no_GSS_top_centrality_normalized' ==  method_name or 'BOIM_fourier_no_GSS_top'== method_name:
                    result = method_details['func'](G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance)
                elif 'BOIM_fourier_no_filtering' in method_name or 'BOIM_vanilla' in method_name:
                    result = method_details['func'](G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources)
                else:
                    result = method_details['func'](G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters)
            elif method_name == 'greedy' or method_name == 'celf' or method_name == 'celfpp':
                result = method_details['func'](G, config, number_of_sources, num_of_sims, diffusion_model)
            elif method_name == 'IMM':
                result = method_details['func'](G, config, number_of_sources, diffusion_model)
            else:
                result = method_details['func'](G, config, number_of_sources)

            if diffusion_model == 'ic':
                spread = effectIC(G, config, result, rounds = num_of_sims)
                print("spread",spread)
            elif diffusion_model == 'lt':
                spread = effectLT(G, config, result, rounds = num_of_sims)
            
            results[method_name].append(spread[0])
            end_time = time.time()
            runtime_results[method_name].append(end_time - start_time)

    print('============================================')
    # print('results:', results)
    for method_name, scores in results.items():
        print('scores:', scores)
        mean_spread = s.mean(scores)
        std_spread = s.stdev(scores) if len(scores) > 1 else 0
        mean_runtime = s.mean(runtime_results[method_name])
        std_runtime = s.stdev(runtime_results[method_name]) if len(runtime_results[method_name]) > 1 else 0
        print(f'{method_name} eval: {mean_spread} +- {std_spread}, runtime: {mean_runtime}s +- {std_runtime}s')

