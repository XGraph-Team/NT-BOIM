from my_util import *

################################ 

# Gaussian Process 
# Fourier Transfer
# filtering through distance，
# KMeans
def BOIM_fourier(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):

    nl = nx.normalized_laplacian_matrix(G)
    _, eig_vect = np.linalg.eigh(nl.todense())
    UT = np.linalg.inv(eig_vect)
    UT_inv = eig_vect

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    #### ####
    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets ,UT)

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_

    groups = [[] for i in range(number_of_clusters)]

    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])
    #### ####

    train_X = []
    train_Y = []

    for i in range(number_of_clusters):
        # randomly select one signal from each cluster
        # TODO, hybrid selection?
        selected_signal = random.choice(groups[i])
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)
    print("train_Y", train_Y)


    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]
    # print('max_train_Y_values', max_train_Y_values)

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(number_of_clusters):

            samples = random.sample(groups[i], 1)
            for sample in samples:
                inputs.append(torch.FloatTensor(sample))

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])
        print("new_Y", new_Y)

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation
        # print("train_Y", train_Y)
        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best = float('-inf')
    identified_signal = None
    print('best_index', best)
    print('train_Y max', train_Y.max().item())

    for signal in sets_after_fourier_transfer:

        input = torch.FloatTensor([signal])
        y_pred = model(input).loc
        # print('y_pred', y_pred)
        # print('best_index', best)
        if y_pred > best:
            best = y_pred
            identified_signal = signal

    identified_set = find_source_set_from_fourier(identified_signal, number_of_sources, UT_inv)
    print("identified_set", identified_set)
    return identified_set

# KMeans
def BOIM_fourier_cuda(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    nl = nx.normalized_laplacian_matrix(G)
    _, eig_vect = np.linalg.eigh(nl.todense())
    UT = np.linalg.inv(eig_vect)
    UT_inv = eig_vect

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets, UT)

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_

    groups = [[] for _ in range(number_of_clusters)]
    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])

    def process_cluster(i):
        selected_signal = random.choice(groups[i])
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)
        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
        return torch.FloatTensor(selected_signal), torch.tensor([[float(e)]])

    results = Parallel(n_jobs=-1)(delayed(process_cluster)(i) for i in range(number_of_clusters))

    train_X = torch.stack([res[0] for res in results]).to(device)
    train_Y = torch.cat([res[1] for res in results]).to(device)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        inputs = []
        for i in range(number_of_clusters):
            samples = random.sample(groups[i], 1)
            for sample in samples:
                inputs.append(torch.FloatTensor(sample))

        inputs = torch.stack(inputs).to(device)

        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=inputs
        )

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]]).to(device)

        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best = float('-inf')
    identified_signal = None

    for signal in sets_after_fourier_transfer:
        input = torch.FloatTensor([signal]).to(device)
        y_pred = model(input).loc
        if y_pred > best:
            best = y_pred
            identified_signal = signal

    identified_set = find_source_set_from_fourier(identified_signal, number_of_sources, UT_inv)

    return identified_set

def BOIM_fourier_debug(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    # Normalized Laplacian L = D^(-1/2) * L * D^(-1/2)
    L = nx.normalized_laplacian_matrix(G)
    UT, UT_inv = compute_eigen_decomposition(L)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    
    # print("candidate_sets", len(candidate_sets)) # 18911
    # sys.exit()
    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets, UT)
    
    groups = perform_clustering_and_select_signals(number_of_clusters, sets_after_fourier_transfer)
    
    train_X, train_Y = simulate_initial_training(groups, G, config, num_of_sims, diffusion_model, number_of_sources, UT_inv)

    train_X, train_Y, function_values, acquisition_values, max_train_Y_values, model = iterative_optimization(train_X, train_Y, groups, number_of_sources, UT_inv, G, config, num_of_sims, diffusion_model, num_iterations)

    identified_set = identify_best_signal(sets_after_fourier_transfer, model, number_of_sources, UT_inv)
    
    return identified_set

def BOIM_fourier_non_normalized(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    # Non-normalized Laplacian L = D - A
    L = nx.laplacian_matrix(G)
    UT, UT_inv = compute_eigen_decomposition(L)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets, UT)
    
    groups = perform_clustering_and_select_signals(number_of_clusters, sets_after_fourier_transfer)

    train_X, train_Y = simulate_initial_training(groups, G, config, num_of_sims, diffusion_model, number_of_sources, UT_inv)

    train_X, train_Y, function_values, acquisition_values, max_train_Y_values, model = iterative_optimization(train_X, train_Y, groups, number_of_sources, UT_inv, G, config, num_of_sims, diffusion_model, num_iterations)

    identified_set = identify_best_signal(sets_after_fourier_transfer, model, number_of_sources, UT_inv)
    
    return identified_set

# Gaussian Process 
# Fourier Transfer
# Filtering through distance，
# RBF kernel
# No GSS
def BOIM_fourier_no_GSS(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance):

    nl = nx.normalized_laplacian_matrix(G)
    _, eig_vect = np.linalg.eigh(nl.todense())
    UT = np.linalg.inv(eig_vect)
    UT_inv = eig_vect

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    train_X = []
    train_Y = []

    for i in range(num_initial_samples):
        # randomly select
        # TODO, hybrid selection?
        selected_set = random.sample(candidate_sets, 1)[0]
        selected_signal = create_signal_from_source_set(G, selected_set, UT)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
        
        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(num_initial_samples):

            selected_set = random.sample(candidate_sets, 1)[0]
            selected_signal = create_signal_from_source_set(G, selected_set, UT)

            input = torch.FloatTensor(selected_signal)
            inputs.append(input)

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    # final_index = sets_after_fourier_transfer.index(s)
    identified_set = find_source_set_from_fourier(s, number_of_sources, UT_inv)

    return identified_set

# FIXME: the name should be hybrid instead of top
def BOIM_fourier_no_GSS_top(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance):

    nl = nx.normalized_laplacian_matrix(G)
    _, eig_vect = np.linalg.eigh(nl.todense())
    UT = np.linalg.inv(eig_vect)
    UT_inv = eig_vect

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidates = deg[:candidate_size]

    train_X = []
    train_Y = []

    top_k = 20  # Number of top-ranked candidates to always include
    random_k = 3  # Number of additional random candidates
    initial_sample_size = top_k + random_k

    # Initial sample selection: top-ranked and random
    top_indices = list(range(top_k))
    random_indices = random.sample(range(top_k, len(candidate_sets)), random_k)
    initial_indices = top_indices + random_indices

    for index in initial_indices:
        selected_set = candidate_sets[index]
        selected_signal = create_signal_from_source_set(G, selected_set, UT)

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)
        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        inputs = []

        # Sampling strategy: include top-ranked and random samples
        top_indices = list(range(top_k))
        random_indices = random.sample(range(top_k, len(candidate_sets)), random_k)
        sample_indices = top_indices + random_indices

        for index in sample_indices:
            selected_set = candidate_sets[index]
            selected_signal = create_signal_from_source_set(G, selected_set, UT)

            input = torch.FloatTensor(selected_signal)
            inputs.append(input)

        inputs = torch.stack(inputs).type(torch.float)

        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()
        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1, 1)], dim=0)

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best_index = train_Y.argmax().item()
    best_signal = train_X[best_index]
    identified_set = find_source_set_from_fourier(best_signal.tolist(), number_of_sources, UT_inv)

    return identified_set

# FIXME: the name should be hybrid instead of top
def BOIM_fourier_no_GSS_top_centrality_normalized(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance):

    # Compute the normalized Laplacian matrix and its eigen decomposition
    nl = nx.normalized_laplacian_matrix(G)
    _, eig_vect = np.linalg.eigh(nl.todense())
    UT = np.linalg.inv(eig_vect)
    UT_inv = eig_vect

    # Create candidate sets
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    # Enhance candidate selection using centrality measures
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidates = [node for node, _ in deg[:candidate_size]]
    candidates = enhance_candidate_selection_with_centrality_normalized(G, candidates)

    train_X = []
    train_Y = []

    top_k = 20  # Number of top-ranked candidates to always include
    random_k = 3  # Number of additional random candidates
    initial_sample_size = top_k + random_k

    # Initial sample selection: top-ranked and random
    top_indices = list(range(top_k))
    random_indices = random.sample(range(top_k, len(candidate_sets)), random_k)
    initial_indices = top_indices + random_indices

    for index in initial_indices:
        selected_set = candidate_sets[index]
        selected_signal = create_signal_from_source_set(G, selected_set, UT)

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)  # Corrected this line
        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        inputs = []

        # Sampling strategy: include top-ranked and random samples
        top_indices = list(range(top_k))
        random_indices = random.sample(range(top_k, len(candidate_sets)), random_k)
        sample_indices = top_indices + random_indices

        for index in sample_indices:
            selected_set = candidate_sets[index]
            selected_signal = create_signal_from_source_set(G, selected_set, UT)

            input = torch.FloatTensor(selected_signal)  # Corrected this line
            inputs.append(input)

        inputs = torch.stack(inputs).type(torch.float)

        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()
        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1, 1)], dim=0)

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best_index = train_Y.argmax().item()
    best_signal = train_X[best_index]
    identified_set = find_source_set_from_fourier(best_signal.tolist(), number_of_sources, UT_inv)

    return identified_set

# Gaussian Process 
# Fourier Transfer, 
# RBF kernel
def BOIM_fourier_no_filtering(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):

    nl = nx.normalized_laplacian_matrix(G)
    _, eig_vect = np.linalg.eigh(nl.todense())
    UT = np.linalg.inv(eig_vect)
    UT_inv = eig_vect

    candidate_sets = create_candidate_set_pool(G, candidate_size, number_of_sources)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets ,UT)

    train_X = []
    train_Y = []

    for i in range(num_initial_samples):

        selected_set = random.sample(candidate_sets, 1)[0]
        selected_signal = create_signal_from_source_set(G, selected_set, UT)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(num_initial_samples):

            selected_set = random.sample(candidate_sets, 1)[0]
            selected_signal = create_signal_from_source_set(G, selected_set, UT)

            input = torch.FloatTensor(selected_signal)
            inputs.append(input)

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    # final_index = sets_after_fourier_transfer.index(s)
    identified_set = find_source_set_from_fourier(s, number_of_sources, UT_inv)

    return identified_set


################################
def BOIM_vanilla_random_centrality(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection with centrality measures
    candidates = enhance_candidate_selection_with_centrality(G, candidates)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    initial_indices = random.sample(range(len(candidate_sets)), 23)

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_adaptive(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidates = deg[:candidate_size]

    top_indices = range(5)  # Start with a small number of top candidates
    random_indices = random.sample(range(5, len(candidate_sets)), 18)  # Start with random samples
    initial_indices = list(top_indices) + random_indices

    train_X, train_Y, source_sets = [], [], []

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)

        input = [1 if item[0] in source_set else 0 for item in candidates]
        input = torch.tensor(input)
        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    for iteration in range(num_iterations):
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        combs = []

        samples = random.sample(candidate_sets, 23)
        for source_set in samples:
            input = [1 if item[0] in source_set else 0 for item in candidates]
            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=combs
        )

        selected = [candidates[i][0] for i in range(candidate_size) if list(candidate.squeeze().numpy())[i] == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1, 1)], dim=0)

        # Adjust strategy based on intermediate performance
        if new_Y > train_Y.max():
            top_indices = range(10)  # Increase the number of top candidates if improving
        else:
            random_indices = random.sample(range(10, len(candidate_sets)), 18)  # Explore more if not improving

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = [candidates[i][0] for i in range(candidate_size) if int(s[i]) == 1]

    return result

def BOIM_vanilla_hybrid_adaptive_centrality_normalized(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidates = [node for node, _ in deg[:candidate_size]]

    candidates = enhance_candidate_selection_with_centrality_normalized(G, candidates)

    top_indices = range(5)  # Start with a small number of top candidates
    random_indices = random.sample(range(5, len(candidate_sets)), 18)  # Start with random samples
    initial_indices = list(top_indices) + random_indices

    train_X, train_Y, source_sets = [], [], []

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)

        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input)
        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    for iteration in range(num_iterations):
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        combs = []

        samples = random.sample(candidate_sets, 23)
        for source_set in samples:
            input = [1 if item in source_set else 0 for item in candidates]
            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=combs
        )

        selected = [candidates[i] for i in range(candidate_size) if list(candidate.squeeze().numpy())[i] == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1, 1)], dim=0)

        # Adjust strategy based on intermediate performance
        if new_Y > train_Y.max():
            top_indices = range(10)  # Increase the number of top candidates if improving
        else:
            random_indices = random.sample(range(10, len(candidate_sets)), 18)  # Explore more if not improving

    best_index = train_Y.argmax().item()
    s = list(train_X[best_index])
    result = [candidates[i] for i in range(candidate_size) if int(s[i]) == 1]

    return result

def BOIM_vanilla_hybrid_adaptive_centrality_normalized_with_early_stopping(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    # tolerance = 0.001 # Tolerance for early stopping
    tolerance = 0.00000000001
    print("tolerance", tolerance)
    min_iterations = 5
    print("min_iterations", min_iterations)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidates = [node for node, _ in deg[:candidate_size]]

    # candidates = enhance_candidate_selection_with_centrality_normalized(G, candidates)

    top_indices = range(5)  # Start with a small number of top candidates
    random_indices = random.sample(range(5, len(candidate_sets)), 18)  # Start with random samples
    initial_indices = list(top_indices) + random_indices

    train_X, train_Y, source_sets = [], [], []

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)

        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input)
        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    prev_max_train_Y = train_Y.max().item()

    for iteration in range(num_iterations):
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        combs = []

        samples = random.sample(candidate_sets, 23)
        for source_set in samples:
            input = [1 if item in source_set else 0 for item in candidates]
            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=combs
        )

        selected = [candidates[i] for i in range(candidate_size) if list(candidate.squeeze().numpy())[i] == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1, 1)], dim=0)

        # Early stopping based on convergence
        max_train_Y = train_Y.max().item()
        print('max_train_Y', max_train_Y)
        print('prev_max_train_Y', prev_max_train_Y)
        if iteration >= min_iterations and abs(max_train_Y - prev_max_train_Y) < tolerance:
            print(f'Early stopping at iteration {iteration} with max influence spread {max_train_Y}')
            break
        prev_max_train_Y = max_train_Y

        # Adjust strategy based on intermediate performance
        if new_Y > train_Y.max():
            top_indices = range(10)  # Increase the number of top candidates if improving
        else:
            random_indices = random.sample(range(10, len(candidate_sets)), 18)  # Explore more if not improving

    best_index = train_Y.argmax().item()
    s = list(train_X[best_index])
    result = [candidates[i] for i in range(candidate_size) if int(s[i]) == 1]

    return result

def BOIM_vanilla_top_no_gp(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    
    candidate_nodes = [node for node, _ in deg[:candidate_size]]  # Extract nodes from tuples

    # Enhance candidate selection with centrality measures
    candidates = enhance_candidate_selection_with_centrality_normalized(G, candidate_nodes)

    train_X = []
    train_Y = []
    source_sets = []

    top_indices = range(23)  # Always select top 23, get the max e from the 23 results.
    # top_indices = range(5)  # Always select top 5, get the max e from the 5 results.

    # top_indices = range(40)  # Always select top 40, get the max e from the 40 results.

    initial_indices = list(top_indices)

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
            print('e', e)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    max_train_Y_values = [train_Y.max().item()]
    # print('max_train_Y_values', max_train_Y_values)

    best_index = train_Y.argmax().item()
    print('best_index', best_index)
    print('train_Y max', train_Y.max().item())
    s = list(train_X[best_index])
    result = [candidates[i] for i in range(candidate_size) if int(s[i]) == 1]
    print('result', result)
    return result

def BOIM_vanilla_random_centrality_normalized(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection with centrality measures
    candidates = enhance_candidate_selection_with_centrality_normalized(G, candidates)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    initial_indices = random.sample(range(len(candidate_sets)), 23)

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result



def BOIM_vanilla_random_L(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    # Compute the normalized Laplacian
    L = nx.normalized_laplacian_matrix(G).astype(float)

    # Compute top two eigenvectors
    _, eigenvectors = compute_spectral_components(L, num_vectors=2) 
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection    
    # Use the Fiedler vector (2nd smallest eigenvalue) to rank nodes
    candidates = enhance_candidate_selection_with_spectra(G, candidates, eigenvectors)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    initial_indices = random.sample(range(len(candidate_sets)), 23)

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

################################ 
# Gaussian Process
# node selection
# RBF kernel
def BOIM_vanilla_random(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)

    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)

    candidates = deg[:candidate_size]
    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    initial_indices = random.sample(range(len(candidate_sets)), 23)

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item[0] in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)
    # print('train_Y', train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = []
            for item in candidates:
                if item[0] in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])
        # print('new_Y', new_Y)

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i][0])
    
    # result = [45, 275, 785]
    # print('result', result)

    return result

from collections import defaultdict

def BOIM_vanilla_stratified(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    num_candidate_sets = len(candidate_sets)

    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidates = deg[:candidate_size]

    # Stratify candidate sets
    strata = stratify_candidate_sets(candidate_sets)
    num_strata = len(strata)
    # print('num_strata', num_strata)

    train_X = []
    train_Y = []
    source_sets = []

    # Initial stratified sampling
    initial_sample_size = 23
    samples_per_stratum = max(1, initial_sample_size // num_strata)
    # print('samples_per_stratum', samples_per_stratum)
    
    for stratum in strata.values():
        # print('stratum', stratum)
        stratum_samples = np.random.choice(stratum, size=min(samples_per_stratum, len(stratum)), replace=False)
        # print('stratum_samples', stratum_samples)
        for index in stratum_samples:
            source_set = candidate_sets[index]

            if diffusion_model == 'ic':
                e, _ = effectIC(G, config, source_set, num_of_sims)
            elif diffusion_model == 'lt':
                e, _ = effectLT(G, config, source_set, num_of_sims)
            else:
                raise NotImplementedError("Diffusion model not recognized.")
                
            input = [1 if item[0] in source_set else 0 for item in candidates]
            input = torch.tensor(input, dtype=torch.double)

            train_X.append(input)
            train_Y.append([float(e)])
            source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        # Stratified sampling for each iteration
        sample_size = 23
        samples_per_stratum = max(1, sample_size // num_strata)
        samples = []
        
        for stratum in strata.values():
            stratum_samples = np.random.choice(stratum, size=min(samples_per_stratum, len(stratum)), replace=False)
            samples.extend([candidate_sets[i] for i in stratum_samples])

        combs = [torch.tensor([1 if item[0] in source_set else 0 for item in candidates], dtype=torch.double) 
                 for source_set in samples]
        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=combs)

        selected = [candidates[i][0] for i in range(candidate_size) if candidate.squeeze()[i] == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)

        train_X = torch.cat([train_X, candidate[0].unsqueeze(0)], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    best_index = train_Y.argmax().item()
    result = [candidates[i][0] for i in range(candidate_size) if train_X[best_index][i] == 1]

    return result

def BOIM_vanilla_stratified_centrality_normalized(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    num_candidate_sets = len(candidate_sets)

    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidates = [node for node, _ in deg[:candidate_size]]

    # Enhance candidate selection with centrality measures
    candidates = enhance_candidate_selection_with_centrality_normalized(G, candidates)

    # Stratify candidate sets
    strata = stratify_candidate_sets(candidate_sets)
    num_strata = len(strata)

    train_X = []
    train_Y = []
    source_sets = []

    initial_sample_size = 23
    samples_per_stratum = max(1, initial_sample_size // num_strata)
    
    for stratum in strata.values():
        stratum_samples = np.random.choice(stratum, size=min(samples_per_stratum, len(stratum)), replace=False)
        for index in stratum_samples:
            source_set = candidate_sets[index]

            if diffusion_model == 'ic':
                e, _ = effectIC(G, config, source_set, num_of_sims)
            elif diffusion_model == 'lt':
                e, _ = effectLT(G, config, source_set, num_of_sims)
            else:
                raise NotImplementedError("Diffusion model not recognized.")
                
            input = [1 if item in source_set else 0 for item in candidates]
            input = torch.tensor(input, dtype=torch.double)

            train_X.append(input)
            train_Y.append([float(e)])
            source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        sample_size = 23
        samples_per_stratum = max(1, sample_size // num_strata)
        samples = []
        
        for stratum in strata.values():
            stratum_samples = np.random.choice(stratum, size=min(samples_per_stratum, len(stratum)), replace=False)
            samples.extend([candidate_sets[i] for i in stratum_samples])

        combs = [torch.tensor([1 if item in source_set else 0 for item in candidates], dtype=torch.double) 
                 for source_set in samples]
        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=combs)

        selected = [candidates[i] for i in range(candidate_size) if candidate.squeeze()[i] == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)

        train_X = torch.cat([train_X, candidate[0].unsqueeze(0)], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    best_index = train_Y.argmax().item()
    result = [candidates[i] for i in range(candidate_size) if train_X[best_index][i] == 1]

    return result

def stratify_candidate_sets(candidate_sets):
    # Stratify based on set size
    strata = defaultdict(list)
    for i, s in enumerate(candidate_sets):
        strata[len(s)].append(i)
    
    # If there = too many strata, group them
    if len(strata) > 5:
        grouped_strata = defaultdict(list)
        for size, indices in strata.items():
            group = size // 5  # Group sizes by 5
            grouped_strata[group].extend(indices)
        return grouped_strata

    # print("Strata:", strata)
    
    return strata

def BOIM_vanilla_cluster(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    num_candidate_sets = len(candidate_sets)

    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidates = deg[:candidate_size]

    # Cluster candidate sets
    clusters = cluster_candidate_sets(candidate_sets)
    num_clusters = len(clusters)
    # print('num_clusters', num_clusters)
    # exit()

    train_X = []
    train_Y = []
    source_sets = []

    # Initial cluster sampling
    initial_sample_size = 23
    clusters_to_sample = min(num_clusters, max(1, initial_sample_size // num_clusters))
    # print('clusters_to_sample', clusters_to_sample)

    if clusters_to_sample > num_clusters:
        clusters_to_sample = num_clusters
        # print('num_clusters', num_clusters)
        # print('clusters_to_sample', clusters_to_sample)
        # exit()
    
    selected_clusters = np.random.choice(list(clusters.keys()), size=clusters_to_sample, replace=False)
    # print(selected_clusters)
    initial_samples = [index for cluster in selected_clusters for index in clusters[cluster]]
    initial_samples = np.random.choice(initial_samples, size=min(initial_sample_size, len(initial_samples)), replace=False)

    for index in initial_samples:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = [1 if item[0] in source_set else 0 for item in candidates]
        input = torch.tensor(input, dtype=torch.double)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        # Cluster sampling for each iteration
        sample_size = 23
        clusters_to_sample = min(num_clusters, max(1, sample_size // num_clusters))
        
        if clusters_to_sample > num_clusters:
            clusters_to_sample = num_clusters
        
        selected_clusters = np.random.choice(list(clusters.keys()), size=clusters_to_sample, replace=False)
        samples = [index for cluster in selected_clusters for index in clusters[cluster]]
        samples = np.random.choice(samples, size=min(sample_size, len(samples)), replace=False)

        combs = [torch.tensor([1 if item[0] in candidate_sets[i] else 0 for item in candidates], dtype=torch.double) 
                 for i in samples]
        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=combs)

        selected = [candidates[i][0] for i in range(candidate_size) if candidate.squeeze()[i] == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)

        train_X = torch.cat([train_X, candidate[0].unsqueeze(0)], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    best_index = train_Y.argmax().item()
    result = [candidates[i][0] for i in range(candidate_size) if train_X[best_index][i] == 1]

    return result

def BOIM_vanilla_cluster_centrality_normalized(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    num_candidate_sets = len(candidate_sets)

    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidates = deg[:candidate_size]

    # Enhance candidate selection with centrality measures
    candidates = enhance_candidate_selection_with_centrality_normalized(G, [node[0] for node in candidates])

    # Cluster candidate sets
    clusters = cluster_candidate_sets(candidate_sets)
    num_clusters = len(clusters)

    train_X = []
    train_Y = []
    source_sets = []

    # Initial cluster sampling
    initial_sample_size = 23
    clusters_to_sample = min(num_clusters, max(1, initial_sample_size // num_clusters))

    if clusters_to_sample > num_clusters:
        clusters_to_sample = num_clusters
    
    selected_clusters = np.random.choice(list(clusters.keys()), size=clusters_to_sample, replace=False)
    initial_samples = [index for cluster in selected_clusters for index in clusters[cluster]]
    initial_samples = np.random.choice(initial_samples, size=min(initial_sample_size, len(initial_samples)), replace=False)

    for index in initial_samples:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input, dtype=torch.double)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        # Cluster sampling for each iteration
        sample_size = 23
        clusters_to_sample = min(num_clusters, max(1, sample_size // num_clusters))
        
        if clusters_to_sample > num_clusters:
            clusters_to_sample = num_clusters
        
        selected_clusters = np.random.choice(list(clusters.keys()), size=clusters_to_sample, replace=False)
        samples = [index for cluster in selected_clusters for index in clusters[cluster]]
        samples = np.random.choice(samples, size=min(sample_size, len(samples)), replace=False)

        combs = [torch.tensor([1 if item in candidate_sets[i] else 0 for item in candidates], dtype=torch.double) 
                 for i in samples]
        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=combs)

        selected = [candidates[i] for i in range(candidate_size) if candidate.squeeze()[i] == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)

        train_X = torch.cat([train_X, candidate[0].unsqueeze(0)], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    best_index = train_Y.argmax().item()
    result = [candidates[i] for i in range(candidate_size) if train_X[best_index][i] == 1]

    return result

def cluster_candidate_sets(candidate_sets):
    # Cluster based on set size 
    clusters = defaultdict(list)
    for i, s in enumerate(candidate_sets):
        clusters[len(s) // 5].append(i)  # Group by size

    print("Clusters:", clusters)
    
    return clusters

def BOIM_vanilla_normal(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    num_candidate_sets = len(candidate_sets)

    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidates = deg[:candidate_size]

    # Adjust mean and standard deviation calculation
    mean = (num_candidate_sets - 1) / 2
    std_dev = num_candidate_sets / 6  

    train_X = []
    train_Y = []
    source_sets = []

    # Improved initial selection
    initial_sample_size = 23
    initial_indices = set()
    while len(initial_indices) < initial_sample_size:
        new_index = int(np.clip(np.random.normal(mean, std_dev), 0, num_candidate_sets - 1))
        initial_indices.add(new_index)

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = [1 if item[0] in source_set else 0 for item in candidates]
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        # Improved Gaussian sampling
        sample_size = 23
        indices = set()
        while len(indices) < sample_size:
            new_index = int(np.clip(np.random.normal(mean, std_dev), 0, num_candidate_sets - 1))
            indices.add(new_index)

        samples = [candidate_sets[i] for i in indices]
        combs = [torch.tensor([1 if item[0] in source_set else 0 for item in candidates]) 
                 for source_set in samples]
        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=combs)

        selected = [candidates[i][0] for i in range(candidate_size) if candidate.squeeze()[i] == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]])

        train_X = torch.cat([train_X, candidate[0].unsqueeze(0)], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    best_index = train_Y.argmax().item()
    result = [candidates[i][0] for i in range(candidate_size) if train_X[best_index][i] == 1]

    return result

def BOIM_vanilla_normal_centrality_normalized(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    num_candidate_sets = len(candidate_sets)

    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidate_nodes = [node for node, _ in deg[:candidate_size]]  # Extract nodes from tuples

    # Enhance candidate selection with centrality measures
    candidates = enhance_candidate_selection_with_centrality_normalized(G, candidate_nodes)

    # Adjust mean and standard deviation calculation
    mean = (num_candidate_sets - 1) / 2
    std_dev = num_candidate_sets / 6  

    train_X = []
    train_Y = []
    source_sets = []

    # Improved initial selection
    initial_sample_size = 23
    initial_indices = set()
    while len(initial_indices) < initial_sample_size:
        new_index = int(np.clip(np.random.normal(mean, std_dev), 0, num_candidate_sets - 1))
        initial_indices.add(new_index)

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        # Improved Gaussian sampling
        sample_size = 23
        indices = set()
        while len(indices) < sample_size:
            new_index = int(np.clip(np.random.normal(mean, std_dev), 0, num_candidate_sets - 1))
            indices.add(new_index)

        samples = [candidate_sets[i] for i in indices]
        combs = [torch.tensor([1 if item in source_set else 0 for item in candidates]) 
                 for source_set in samples]
        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=combs)

        selected = [candidates[i] for i in range(candidate_size) if candidate.squeeze()[i] == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]])

        train_X = torch.cat([train_X, candidate[0].unsqueeze(0)], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    best_index = train_Y.argmax().item()
    result = [candidates[i] for i in range(candidate_size) if train_X[best_index][i] == 1]

    return result

def BOIM_vanilla_top(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)

    candidates = deg[:candidate_size]
    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    top_indices = range(23)  # Always select top 5

    initial_indices = list(top_indices)

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = []
        for item in candidates:
            if item[0] in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = []
            for item in candidates:
                if item[0] in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i][0])

    return result

def BOIM_vanilla_top_centrality_normalized(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    
    candidate_nodes = [node for node, _ in deg[:candidate_size]]  # Extract nodes from tuples

    # Enhance candidate selection with centrality measures
    candidates = enhance_candidate_selection_with_centrality_normalized(G, candidate_nodes)

    train_X = []
    train_Y = []
    source_sets = []

    top_indices = range(3)  # Always select top 23

    initial_indices = list(top_indices)

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]
    # print('max_train_Y_values', max_train_Y_values)

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = [1 if item in source_set else 0 for item in candidates]
            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,  # Number of candidates to sample in each iteration
            choices=combs)

        selected = [candidates[i] for i in range(candidate_size) if candidate.squeeze()[i] == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
            print('e', e)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])
        print("new_Y", new_Y)

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1, 1)], dim=0)  # Add a new dimension for the new evaluation
        # print("train_X", train_X)
        print("train_Y 2", train_Y)
        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    best_index = train_Y.argmax().item()
    print('best_index', best_index)
    print('train_Y max', train_Y.max().item())
    s = list(train_X[best_index])
    # print('s', s)
    result = [candidates[i] for i in range(candidate_size) if int(s[i]) == 1]
    print('result', result)
    return result

def BOIM_vanilla_hybrid(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    
    # print("candidate_sets", candidate_sets)
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    # print('deg', deg)
    candidates = deg[:candidate_size]
    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    # initial_indices = random.sample(range(len(candidate_sets)), 23)

    # Zijian: 
    # Use a combination of deterministic and random elements. 
    # For example, always include a few top-ranked candidate sets,
    # and fill the remaining slots with randomly selected sets. 
    # This approach balances exploration (randomness) and exploitation 
    # (using known good candidates).
    top_indices = range(3)  # Always select top 3
    # print("top_indices", top_indices)
    random_indices = random.sample(range(3, len(candidate_sets)), 20)
    # print("random_indices", random_indices)
    initial_indices = list(top_indices) + random_indices

    # print("initial_indices", initial_indices)

    for index in initial_indices:
        source_set = candidate_sets[index]
        # print("index", index)
        # print("source_set", source_set)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item[0] in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = []
            for item in candidates:
                if item[0] in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i][0])

    return result

def BOIM_vanilla_hybrid_centrality(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection with centrality measures
    candidates = enhance_candidate_selection_with_centrality(G, candidates)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    # initial_indices = random.sample(range(len(candidate_sets)), 23)

    # Zijian: 
    # Use a combination of deterministic and random elements. 
    # For example, always include a few top-ranked candidate sets,
    # and fill the remaining slots with randomly selected sets. 
    # This approach balances exploration (randomness) and exploitation 
    # (using known good candidates).
    top_indices = range(3)  # Always select top 3
    random_indices = random.sample(range(3, len(candidate_sets)), 20)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)
        for source_set in samples:
            input = []

            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_nnl(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):

    # Compute the non-normalized Laplacian
    L = nx.laplacian_matrix(G)

    # Compute top two eigenvectors
    _, eigenvectors = compute_spectral_components(L, num_vectors=2) 
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection
    candidates = enhance_candidate_selection_with_spectra(G, candidates, eigenvectors)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    # initial_indices = random.sample(range(len(candidate_sets)), 23)

    # Zijian: 
    # Use a combination of deterministic and random elements. 
    # For example, always include a few top-ranked candidate sets,
    # and fill the remaining slots with randomly selected sets. 
    # This approach balances exploration (randomness) and exploitation 
    # (using known good candidates).
    top_indices = range(3)  # Always select top 3
    random_indices = random.sample(range(3, len(candidate_sets)), 20)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_L(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    # Compute the normalized Laplacian
    L = nx.normalized_laplacian_matrix(G).astype(float)

    # Compute top two eigenvectors
    _, eigenvectors = compute_spectral_components(L, num_vectors=2) 
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection    
    # Use the Fiedler vector (2nd smallest eigenvalue) to rank nodes
    candidates = enhance_candidate_selection_with_spectra(G, candidates, eigenvectors)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    # initial_indices = random.sample(range(len(candidate_sets)), 23)

    # Zijian: 
    # Use a combination of deterministic and random elements. 
    # For example, always include a few top-ranked candidate sets,
    # and fill the remaining slots with randomly selected sets. 
    # This approach balances exploration (randomness) and exploitation 
    # (using known good candidates).
    top_indices = range(3)  # Always select top 3
    random_indices = random.sample(range(3, len(candidate_sets)), 20)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_tune(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)

    candidates = deg[:candidate_size]
    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    # initial_indices = random.sample(range(len(candidate_sets)), 23)

    # Zijian: 
    # Use a combination of deterministic and random elements. 
    # For example, always include a few top-ranked candidate sets,
    # and fill the remaining slots with randomly selected sets. 
    # This approach balances exploration (randomness) and exploitation 
    # (using known good candidates).

    top_k = 20
    random_k = 3  
    top_indices = range(top_k)  
    random_indices = random.sample(range(top_k, len(candidate_sets)), random_k)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]
        # print("index", index)
        # print("source_set", source_set)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item[0] in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # Use the total number of samples from top_k + random_k
        ################################################
        combs = []

        samples = random.sample(candidate_sets, top_k + random_k)  # Use top_k + random_k

        for source_set in samples:
            input = []
            for item in candidates:
                if item[0] in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i][0])

    return result

def BOIM_vanilla_hybrid_tune_L(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    # Compute the normalized Laplacian
    L = nx.normalized_laplacian_matrix(G).astype(float)

    # Compute top two eigenvectors
    _, eigenvectors = compute_spectral_components(L, num_vectors=2) 
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection    
    # Use the Fiedler vector (2nd smallest eigenvalue) to rank nodes
    candidates = enhance_candidate_selection_with_spectra(G, candidates, eigenvectors)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    # initial_indices = random.sample(range(len(candidate_sets)), 23)

    # Zijian: 
    # Use a combination of deterministic and random elements. 
    # For example, always include a few top-ranked candidate sets,
    # and fill the remaining slots with randomly selected sets. 
    # This approach balances exploration (randomness) and exploitation 
    # (using known good candidates).
    top_k = 20
    random_k = 3  
    top_indices = range(top_k)  
    random_indices = random.sample(range(top_k, len(candidate_sets)), random_k)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

       ################################################
        # Use the total number of samples from top_k + random_k
        ################################################
        combs = []

        samples = random.sample(candidate_sets, top_k + random_k)  # Use top_k + random_k

        for source_set in samples:
            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_tune_centrality(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection with centrality measures
    candidates = enhance_candidate_selection_with_centrality(G, candidates)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    # initial_indices = random.sample(range(len(candidate_sets)), 23)

    # Zijian: 
    # Use a combination of deterministic and random elements. 
    # For example, always include a few top-ranked candidate sets,
    # and fill the remaining slots with randomly selected sets. 
    # This approach balances exploration (randomness) and exploitation 
    # (using known good candidates).
    top_k = 20
    random_k = 3  
    top_indices = range(top_k)  
    random_indices = random.sample(range(top_k, len(candidate_sets)), random_k)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # Use the total number of samples from top_k + random_k
        ################################################
        combs = []

        samples = random.sample(candidate_sets, top_k + random_k)  # Use top_k + random_k

        for source_set in samples:
            input = []



            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_tune_centrality_normalized(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection with centrality measures
    candidates = enhance_candidate_selection_with_centrality_normalized(G, candidates)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    # initial_indices = random.sample(range(len(candidate_sets)), 23)

    # Zijian: 
    # Use a combination of deterministic and random elements. 
    # For example, always include a few top-ranked candidate sets,
    # and fill the remaining slots with randomly selected sets. 
    # This approach balances exploration (randomness) and exploitation 
    # (using known good candidates).
    top_k = 20
    random_k = 3  
    top_indices = range(top_k)  
    random_indices = random.sample(range(top_k, len(candidate_sets)), random_k)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # Use the total number of samples from top_k + random_k
        ################################################
        combs = []

        samples = random.sample(candidate_sets, top_k + random_k)  # Use top_k + random_k

        for source_set in samples:
            input = []



            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_tune_degree_betweenness_closeness_eigenvector(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection with centrality measures
    candidates = enhance_candidate_selection_with_degree_betweenness_closeness_eigenvector(G, candidates)

    # Initialize the GP model with several (c, s) pairs
    train_X = []
    train_Y = []
    source_sets = []

    # Use a combination of deterministic and random elements
    top_k = 20
    random_k = 3  
    top_indices = range(top_k)  
    random_indices = random.sample(range(top_k, len(candidate_sets)), random_k)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        # Use the total number of samples from top_k + random_k
        combs = []
        samples = random.sample(candidate_sets, top_k + random_k)  # Use top_k + random_k

        for source_set in samples:
            input = [1 if item in source_set else 0 for item in candidates]
            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,  # Number of candidates to sample in each iteration
            choices=combs
        )

        selected = [candidates[i] for i in range(candidate_size) if list(candidate.squeeze().numpy())[i] == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1, 1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = [candidates[i] for i in range(candidate_size) if int(s[i]) == 1]

    return result

def BOIM_vanilla_hybrid_tune_centrality_threads(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    candidates = enhance_candidate_selection_with_centrality(G, candidates)

    train_X = []
    train_Y = []
    source_sets = []

    top_k = 20
    random_k = 3  
    top_indices = range(top_k)  
    random_indices = random.sample(range(top_k, len(candidate_sets)), random_k)
    initial_indices = list(top_indices) + random_indices

    def evaluate_source_set(source_set):
        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
        return e

    with ThreadPoolExecutor() as executor:
        initial_results = list(executor.map(evaluate_source_set, [candidate_sets[i] for i in initial_indices]))

    for i, index in enumerate(initial_indices):
        source_set = candidate_sets[index]
        e = initial_results[i]

        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        combs = []
        samples = random.sample(candidate_sets, top_k + random_k)

        for source_set in samples:
            input = [1 if item in source_set else 0 for item in candidates]
            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(acq_function=acq_func, q=1, choices=combs)

        selected = [candidates[i] for i in range(candidate_size) if candidate.squeeze().numpy()[i] == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)

        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = [candidates[i] for i in range(candidate_size) if int(s[i]) == 1]

    return result

def BOIM_vanilla_hybrid_tune_centrality_process(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    candidates = enhance_candidate_selection_with_centrality(G, candidates)

    train_X = []
    train_Y = []
    source_sets = []

    top_k = 20
    random_k = 3  
    top_indices = range(top_k)  
    random_indices = random.sample(range(top_k, len(candidate_sets)), random_k)
    initial_indices = list(top_indices) + random_indices

    with ProcessPoolExecutor() as executor:
        initial_results = list(executor.map(
            evaluate_source, 
            [candidate_sets[i] for i in initial_indices], 
            [G] * len(initial_indices), 
            [config] * len(initial_indices), 
            [num_of_sims] * len(initial_indices), 
            [diffusion_model] * len(initial_indices)
        ))

    for i, index in enumerate(initial_indices):
        source_set = candidate_sets[index]
        e = initial_results[i]

        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        combs = []
        samples = random.sample(candidate_sets, top_k + random_k)

        for source_set in samples:
            input = [1 if item in source_set else 0 for item in candidates]
            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(acq_function=acq_func, q=1, choices=combs)

        selected = [candidates[i] for i in range(candidate_size) if candidate.squeeze().numpy()[i] == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1, 1)], dim=0)

        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = [candidates[i] for i in range(candidate_size) if int(s[i]) == 1]

    return result

def BOIM_vanilla_hybrid_tune_centrality_dynamic(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection
    candidates = enhance_candidate_selection_with_centrality(G, candidates)

    # Initialize the GP model with several (c, s) pairs
    train_X = []
    train_Y = []
    source_sets = []

    # Adaptive combination of deterministic and random elements
    top_k = 20 
    random_k = 3 
    top_indices = range(top_k)
    random_indices = random.sample(range(top_k, len(candidate_sets)), random_k)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # Use the total number of samples from top_k + random_k
        ################################################
        combs = []

        samples = random.sample(candidate_sets, top_k + random_k)  # Use top_k + random_k

        for source_set in samples:
            input = [1 if item in source_set else 0 for item in candidates]
            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,  # Number of candidates to sample in each iteration
            choices=combs)

        selected = [candidates[i] for i in range(candidate_size) if list(candidate.squeeze().numpy())[i] == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_tune_centrality_dynamic_multiGP(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection with centrality measures
    candidates = enhance_candidate_selection_with_centrality(G, candidates)

    # Initialize the GP model with several (c, s) pairs
    train_X = []
    train_Y = []
    source_sets = []

    top_k = 20
    random_k = 3  
    top_indices = range(top_k)
    random_indices = random.sample(range(top_k, len(candidate_sets)), random_k)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    previous_max_performance = train_Y.max().item()
    performance_threshold = 0.05

    for iteration in range(num_iterations):
        # Fit a multi-task GP model to the observed data
        kernel = ScaleKernel(MaternKernel(nu=2.5))
        model = RBFSingleTaskGP(train_X, train_Y, covar_module=kernel)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        combs = []

        ################################################
        # Use the total number of samples from top_k + random_k
        ################################################
        samples = random.sample(candidate_sets, top_k + random_k)

        for source_set in samples:
            input = [1 if item in source_set else 0 for item in candidates]
            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=combs)

        selected = [candidates[i] for i in range(candidate_size) if list(candidate.squeeze().numpy())[i] == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1, 1)], dim=0)

        current_max_performance = train_Y.max().item()
        if current_max_performance > previous_max_performance * (1 + performance_threshold):
            top_k = min(top_k + 1, candidate_size)
            random_k = max(random_k - 1, 1)
        else:
            top_k = max(top_k - 1, 1)
            random_k = min(random_k + 1, candidate_size - top_k)

        previous_max_performance = current_max_performance

        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_tune_centrality_multiGP(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection with centrality measures
    candidates = enhance_candidate_selection_with_centrality(G, candidates)

    # Initialize the GP model with several (c, s) pairs
    train_X = []
    train_Y = []
    source_sets = []

    top_k = 20
    random_k = 3  
    top_indices = range(top_k)
    random_indices = random.sample(range(top_k, len(candidate_sets)), random_k)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-task GP model to the observed data with a Matern kernel
        kernel = ScaleKernel(MaternKernel(nu=2.5))
        model = RBFSingleTaskGP(train_X, train_Y, covar_module=kernel)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # Use the total number of samples from top_k + random_k
        ################################################
        combs = []

        samples = random.sample(candidate_sets, top_k + random_k)

        for source_set in samples:
            input = [1 if item in source_set else 0 for item in candidates]
            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=combs)

        selected = [candidates[i] for i in range(candidate_size) if list(candidate.squeeze().numpy())[i] == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1, 1)], dim=0)

        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_g_to_h_0(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    # Compute the normalized Laplacian
    # L = nx.normalized_laplacian_matrix(G).astype(float)
    L= get_laplacian_0(G)

    # Compute top two eigenvectors
    _, eigenvectors = compute_spectral_components(L, num_vectors=2) 
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection
    candidates = enhance_candidate_selection_with_spectra(G, candidates, eigenvectors)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    # initial_indices = random.sample(range(len(candidate_sets)), 23)

    # Zijian: 
    # Use a combination of deterministic and random elements. 
    # For example, always include a few top-ranked candidate sets,
    # and fill the remaining slots with randomly selected sets. 
    # This approach balances exploration (randomness) and exploitation 
    # (using known good candidates).
    top_indices = range(3)  # Always select top 3
    random_indices = random.sample(range(3, len(candidate_sets)), 20)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_g_to_h_1(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    # Compute the normalized Laplacian
    # L = nx.normalized_laplacian_matrix(G).astype(float)
    L= get_laplacian_1(G)

    # Compute top two eigenvectors
    _, eigenvectors = compute_spectral_components(L, num_vectors=2) 
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection
    candidates = enhance_candidate_selection_with_spectra(G, candidates, eigenvectors)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    # initial_indices = random.sample(range(len(candidate_sets)), 23)

    # Zijian: 
    # Use a combination of deterministic and random elements. 
    # For example, always include a few top-ranked candidate sets,
    # and fill the remaining slots with randomly selected sets. 
    # This approach balances exploration (randomness) and exploitation 
    # (using known good candidates).
    top_indices = range(3)  # Always select top 3
    random_indices = random.sample(range(3, len(candidate_sets)), 20)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_g_to_h_2(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    # Compute the normalized Laplacian
    # L = nx.normalized_laplacian_matrix(G).astype(float)
    L= get_laplacian_2(G)

    # Compute top two eigenvectors
    _, eigenvectors = compute_spectral_components(L, num_vectors=2) 
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection
    candidates = enhance_candidate_selection_with_spectra(G, candidates, eigenvectors)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    # initial_indices = random.sample(range(len(candidate_sets)), 23)

    # Zijian: 
    # Use a combination of deterministic and random elements. 
    # For example, always include a few top-ranked candidate sets,
    # and fill the remaining slots with randomly selected sets. 
    # This approach balances exploration (randomness) and exploitation 
    # (using known good candidates).
    top_indices = range(3)  # Always select top 3
    random_indices = random.sample(range(3, len(candidate_sets)), 20)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_g_to_h_3(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    # Compute the normalized Laplacian
    # L = nx.normalized_laplacian_matrix(G).astype(float)
    L= get_laplacian_3(G)

    # Compute top two eigenvectors
    _, eigenvectors = compute_spectral_components(L, num_vectors=2) 
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection
    candidates = enhance_candidate_selection_with_spectra(G, candidates, eigenvectors)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    # initial_indices = random.sample(range(len(candidate_sets)), 23)

    # Zijian: 
    # Use a combination of deterministic and random elements. 
    # For example, always include a few top-ranked candidate sets,
    # and fill the remaining slots with randomly selected sets. 
    # This approach balances exploration (randomness) and exploitation 
    # (using known good candidates).
    top_indices = range(3)  # Always select top 3
    random_indices = random.sample(range(3, len(candidate_sets)), 20)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_x1_y1_z0(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    # Compute the normalized Laplacian
    simplices = generate_simplices(G, dimension=1)
    hodge = HodgeLaplacians(simplices, maxdimension=1)
    L = hodge.getHodgeLaplacian(0)

    # Compute top two eigenvectors
    _, eigenvectors = compute_spectral_components(L, num_vectors=2) 
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection
    candidates = enhance_candidate_selection_with_spectra(G, candidates, eigenvectors)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    # initial_indices = random.sample(range(len(candidate_sets)), 23)

    # Zijian: 
    # Use a combination of deterministic and random elements. 
    # For example, always include a few top-ranked candidate sets,
    # and fill the remaining slots with randomly selected sets. 
    # This approach balances exploration (randomness) and exploitation 
    # (using known good candidates).
    top_indices = range(3)  # Always select top 3
    random_indices = random.sample(range(3, len(candidate_sets)), 20)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_x1_y1_z1(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    # Compute the normalized Laplacian
    simplices = generate_simplices(G, dimension=1)
    hodge = HodgeLaplacians(simplices, maxdimension=1)
    L = hodge.getHodgeLaplacian(1)

    # Compute top two eigenvectors
    _, eigenvectors = compute_spectral_components(L, num_vectors=2) 
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection
    candidates = enhance_candidate_selection_with_spectra(G, candidates, eigenvectors)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    # initial_indices = random.sample(range(len(candidate_sets)), 23)

    # Zijian: 
    # Use a combination of deterministic and random elements. 
    # For example, always include a few top-ranked candidate sets,
    # and fill the remaining slots with randomly selected sets. 
    # This approach balances exploration (randomness) and exploitation 
    # (using known good candidates).
    top_indices = range(3)  # Always select top 3
    random_indices = random.sample(range(3, len(candidate_sets)), 20)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_x2_y2_z1(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    # Compute the normalized Laplacian
    simplices = generate_simplices(G, dimension=2)
    hodge = HodgeLaplacians(simplices, maxdimension=2)
    L = hodge.getHodgeLaplacian(1)

    # Compute top two eigenvectors
    _, eigenvectors = compute_spectral_components(L, num_vectors=2) 
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection
    candidates = enhance_candidate_selection_with_spectra(G, candidates, eigenvectors)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    # initial_indices = random.sample(range(len(candidate_sets)), 23)

    # Zijian: 
    # Use a combination of deterministic and random elements. 
    # For example, always include a few top-ranked candidate sets,
    # and fill the remaining slots with randomly selected sets. 
    # This approach balances exploration (randomness) and exploitation 
    # (using known good candidates).
    top_indices = range(3)  # Always select top 3
    random_indices = random.sample(range(3, len(candidate_sets)), 20)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_x2_y2_z2(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    # Compute the normalized Laplacian
    simplices = generate_simplices(G, dimension=2)
    hodge = HodgeLaplacians(simplices, maxdimension=2)
    L = hodge.getHodgeLaplacian(2)

    # Compute top two eigenvectors
    _, eigenvectors = compute_spectral_components(L, num_vectors=2) 
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection
    candidates = enhance_candidate_selection_with_spectra(G, candidates, eigenvectors)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    # initial_indices = random.sample(range(len(candidate_sets)), 23)

    # Zijian: 
    # Use a combination of deterministic and random elements. 
    # For example, always include a few top-ranked candidate sets,
    # and fill the remaining slots with randomly selected sets. 
    # This approach balances exploration (randomness) and exploitation 
    # (using known good candidates).
    top_indices = range(3)  # Always select top 3
    random_indices = random.sample(range(3, len(candidate_sets)), 20)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

def BOIM_vanilla_hybrid_x3_y3_z2(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    # Compute the normalized Laplacian
    simplices = generate_simplices(G, dimension=3)
    hodge = HodgeLaplacians(simplices, maxdimension=3)
    L = hodge.getHodgeLaplacian(2)

    # Compute top two eigenvectors
    _, eigenvectors = compute_spectral_components(L, num_vectors=2) 
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    candidates = [node for node, _ in sorted(G.degree, key=lambda x: x[1], reverse=True)[:candidate_size]]
    
    # Enhance candidate selection
    candidates = enhance_candidate_selection_with_spectra(G, candidates, eigenvectors)

    # initialize the GP model with several (c, s) pairs

    train_X = []
    train_Y = []
    source_sets = []

    # initial_indices = random.sample(range(len(candidate_sets)), 23)

    # Zijian: 
    # Use a combination of deterministic and random elements. 
    # For example, always include a few top-ranked candidate sets,
    # and fill the remaining slots with randomly selected sets. 
    # This approach balances exploration (randomness) and exploitation 
    # (using known good candidates).
    top_indices = range(3)  # Always select top 3
    random_indices = random.sample(range(3, len(candidate_sets)), 20)
    initial_indices = list(top_indices) + random_indices

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 23 random samples
        ################################################
        combs = []

        samples = random.sample(candidate_sets, 23)

        for source_set in samples:
            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result


################################ hodgelaplacians degree and dimension

# Format
# BOIM_fourier_hodgelaplacians_X_simplcies_Y_maxdim_Z_degree
# X: The X-simplices (G, the dimension of simplices generated).
# Y: The maximum dimension (maxdimension) used in HodgeLaplacians library.
# Z: The degree for which the Hodge Laplacian = computed.

def BOIM_fourier_hodgelaplacians_1_simplcies_1_maxdim_0_degree(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    
    simplices = generate_simplices(G, dimension=1)
    hodge = HodgeLaplacians(simplices, maxdimension=1)
    L = hodge.getHodgeLaplacian(0)
    UT, UT_inv = compute_eigen_decomposition(L)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets, UT)
    
    groups = perform_clustering_and_select_signals(number_of_clusters, sets_after_fourier_transfer)

    train_X, train_Y = simulate_initial_training(groups, G, config, num_of_sims, diffusion_model, number_of_sources, UT_inv)

    train_X, train_Y, function_values, acquisition_values, max_train_Y_values, model = iterative_optimization(train_X, train_Y, groups, number_of_sources, UT_inv, G, config, num_of_sims, diffusion_model, num_iterations)

    identified_set = identify_best_signal(sets_after_fourier_transfer, model, number_of_sources, UT_inv)
    
    return identified_set

def BOIM_fourier_hodgelaplacians_1_simplcies_1_maxdim_1_degree(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    
    simplices = generate_simplices(G, dimension=1)
    hodge = HodgeLaplacians(simplices, maxdimension=1)
    L = hodge.getHodgeLaplacian(1)
    # print("type of L", type(L))
    UT, UT_inv = compute_eigen_decomposition(L)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets, UT)
    
    groups = perform_clustering_and_select_signals(number_of_clusters, sets_after_fourier_transfer)

    train_X, train_Y = simulate_initial_training(groups, G, config, num_of_sims, diffusion_model, number_of_sources, UT_inv)

    train_X, train_Y, function_values, acquisition_values, max_train_Y_values, model = iterative_optimization(train_X, train_Y, groups, number_of_sources, UT_inv, G, config, num_of_sims, diffusion_model, num_iterations)

    identified_set = identify_best_signal(sets_after_fourier_transfer, model, number_of_sources, UT_inv)
    
    return identified_set

def BOIM_fourier_hodgelaplacians_1_simplcies_1_maxdim_1_degree_mapping(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    
    simplices = generate_simplices(G, dimension=1)
    hodge = HodgeLaplacians(simplices, maxdimension=1)
    L = hodge.getHodgeLaplacian(1)
    UT, UT_inv = compute_eigen_decomposition(L)
    # print("Shape of the UT:", UT.shape)

    candidate_sets = create_candidate_set_pool_filtering_mapping_weighted_edges(G, candidate_size, number_of_sources, allowed_shortest_distance)
    # print("candidate_sets", len(candidate_sets)) # 19600
    # sys.exit()
    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets, UT)
    
    groups = perform_clustering_and_select_signals(number_of_clusters, sets_after_fourier_transfer)
    
    train_X, train_Y = simulate_initial_training(groups, G, config, num_of_sims, diffusion_model, number_of_sources, UT_inv)

    train_X, train_Y, function_values, acquisition_values, max_train_Y_values, model = iterative_optimization(train_X, train_Y, groups, number_of_sources, UT_inv, G, config, num_of_sims, diffusion_model, num_iterations)

    identified_set = identify_best_signal(sets_after_fourier_transfer, model, number_of_sources, UT_inv)
    
    return identified_set

def BOIM_fourier_hodgelaplacians_2_simplcies_2_maxdim_1_degree(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    
    simplices = generate_simplices(G, dimension=2)
    hodge = HodgeLaplacians(simplices, maxdimension=2)
    L = hodge.getHodgeLaplacian(1)

    UT, UT_inv = compute_eigen_decomposition(L)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets, UT)
    
    groups = perform_clustering_and_select_signals(number_of_clusters, sets_after_fourier_transfer)

    train_X, train_Y = simulate_initial_training(groups, G, config, num_of_sims, diffusion_model, number_of_sources, UT_inv)

    train_X, train_Y, function_values, acquisition_values, max_train_Y_values, model = iterative_optimization(train_X, train_Y, groups, number_of_sources, UT_inv, G, config, num_of_sims, diffusion_model, num_iterations)

    identified_set = identify_best_signal(sets_after_fourier_transfer, model, number_of_sources, UT_inv)
    
    return identified_set

def BOIM_fourier_hodgelaplacians_2_simplcies_2_maxdim_2_degree(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    
    simplices = generate_simplices(G, dimension=2)
    hodge = HodgeLaplacians(simplices, maxdimension=2)
    L = hodge.getHodgeLaplacian(2)

    UT, UT_inv = compute_eigen_decomposition(L)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets, UT)
    
    groups = perform_clustering_and_select_signals(number_of_clusters, sets_after_fourier_transfer)

    train_X, train_Y = simulate_initial_training(groups, G, config, num_of_sims, diffusion_model, number_of_sources, UT_inv)

    train_X, train_Y, function_values, acquisition_values, max_train_Y_values, model = iterative_optimization(train_X, train_Y, groups, number_of_sources, UT_inv, G, config, num_of_sims, diffusion_model, num_iterations)

    identified_set = identify_best_signal(sets_after_fourier_transfer, model, number_of_sources, UT_inv)
    
    return identified_set

def BOIM_fourier_hodgelaplacians_3_simplcies_3_maxdim_2_degree(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    
    simplices = generate_simplices(G, dimension=3)
    hodge = HodgeLaplacians(simplices, maxdimension=3)
    L = hodge.getHodgeLaplacian(2)

    UT, UT_inv = compute_eigen_decomposition(L)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets, UT)
    
    groups = perform_clustering_and_select_signals(number_of_clusters, sets_after_fourier_transfer)

    train_X, train_Y = simulate_initial_training(groups, G, config, num_of_sims, diffusion_model, number_of_sources, UT_inv)

    train_X, train_Y, function_values, acquisition_values, max_train_Y_values, model = iterative_optimization(train_X, train_Y, groups, number_of_sources, UT_inv, G, config, num_of_sims, diffusion_model, num_iterations)

    identified_set = identify_best_signal(sets_after_fourier_transfer, model, number_of_sources, UT_inv)
    
    return identified_set

def BOIM_fourier_hodgelaplacians_3_simplcies_3_maxdim_3_degree(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    
    simplices = generate_simplices(G, dimension=3)
    hodge = HodgeLaplacians(simplices, maxdimension=3)
    L = hodge.getHodgeLaplacian(3)

    UT, UT_inv = compute_eigen_decomposition(L)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets, UT)
    
    groups = perform_clustering_and_select_signals(number_of_clusters, sets_after_fourier_transfer)

    train_X, train_Y = simulate_initial_training(groups, G, config, num_of_sims, diffusion_model, number_of_sources, UT_inv)

    train_X, train_Y, function_values, acquisition_values, max_train_Y_values, model = iterative_optimization(train_X, train_Y, groups, number_of_sources, UT_inv, G, config, num_of_sims, diffusion_model, num_iterations)

    identified_set = identify_best_signal(sets_after_fourier_transfer, model, number_of_sources, UT_inv)
    
    return identified_set

################################ hodgelaplacians combined
def BOIM_fourier_hodgelaplacians_combined_0_1_max_dim_1_N_N(G, simplices, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    # test max dimension = 1
    hodge = HodgeLaplacians(simplices, maxdimension=1)  # or higher, depending on the data

    L_0 = hodge.getHodgeLaplacian(d=0)  # Using 0-simplices (nodes)
    # print(hodge_laplacian)  
    hodge = HodgeLaplacians(simplices, maxdimension=2) 
    L_1 = hodge.getHodgeLaplacian(d=1)

    # Get the incidence matrix
    B = nx.incidence_matrix(G, oriented=True)  # Set 'oriented=True' for a directed incidence matrix

    n_n_matrix = get_n_n_from_hodge(L_0, L_1, B)

    # Compute eigenvalues and eigenvectors of the resulting matrix
    eig_values, eig_vect = np.linalg.eigh(n_n_matrix.toarray())  # Make sure to convert to dense if necessary
    UT = np.linalg.inv(eig_vect)  # Inverse of eigenvectors
    UT_inv = eig_vect  # Directly use eigenvectors
    # print("Shape of the UT:", UT.shape)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    # print("candidate_sets", candidate_sets)
    # print("len(candidate_sets)", len(candidate_sets))

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets ,UT)
    # print("sets_after_fourier_transfer", sets_after_fourier_transfer)
    # print("len(sets_after_fourier_transfer)", len(sets_after_fourier_transfer))

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_

    groups = [[] for i in range(number_of_clusters)]

    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])

    train_X = []
    train_Y = []

    for i in range(number_of_clusters):

        selected_signal = random.choice(groups[i])
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)
        # print("source_set", source_set)
        # print("len(source_set)", len(source_set))

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(number_of_clusters):

            samples = random.sample(groups[i], 1)
            for sample in samples:
                inputs.append(torch.FloatTensor(sample))

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best = float('-inf')
    identified_signal = None

    for signal in sets_after_fourier_transfer:

        input = torch.FloatTensor([signal])
        y_pred = model(input).loc
        if y_pred > best:
            best = y_pred
            identified_signal = signal

    identified_set = find_source_set_from_fourier(identified_signal, number_of_sources, UT_inv)

    return identified_set

def BOIM_fourier_hodgelaplacians_combined_0_1_max_dim_2_N_N(G, simplices, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    
    # test max dimension = 2
    hodge = HodgeLaplacians(simplices, maxdimension=2)  # or higher, depending on the data
    L_0 = hodge.getHodgeLaplacian(d=0)  # Using 0-simplices (nodes)
    # print(hodge_laplacian)  
    hodge = HodgeLaplacians(simplices, maxdimension=2) 
    L_1 = hodge.getHodgeLaplacian(d=1)

    # Get the incidence matrix
    B = nx.incidence_matrix(G, oriented=True)  # Set 'oriented=True' for a directed incidence matrix

    n_n_matrix = get_n_n_from_hodge(L_0, L_1, B)

    # Compute eigenvalues and eigenvectors of the resulting matrix
    eig_values, eig_vect = np.linalg.eigh(n_n_matrix.toarray())  # Make sure to convert to dense if necessary
    UT = np.linalg.inv(eig_vect)  # Inverse of eigenvectors
    UT_inv = eig_vect  # Directly use eigenvectors
    # print("Shape of the UT:", UT.shape)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    # print("candidate_sets", candidate_sets)
    # print("len(candidate_sets)", len(candidate_sets))

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets ,UT)
    # print("sets_after_fourier_transfer", sets_after_fourier_transfer)
    # print("len(sets_after_fourier_transfer)", len(sets_after_fourier_transfer))

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_

    groups = [[] for i in range(number_of_clusters)]

    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])

    train_X = []
    train_Y = []

    for i in range(number_of_clusters):

        selected_signal = random.choice(groups[i])
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)
        # print("source_set", source_set)
        # print("len(source_set)", len(source_set))

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(number_of_clusters):

            samples = random.sample(groups[i], 1)
            for sample in samples:
                inputs.append(torch.FloatTensor(sample))

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best = float('-inf')
    identified_signal = None

    for signal in sets_after_fourier_transfer:

        input = torch.FloatTensor([signal])
        y_pred = model(input).loc
        if y_pred > best:
            best = y_pred
            identified_signal = signal

    identified_set = find_source_set_from_fourier(identified_signal, number_of_sources, UT_inv)

    return identified_set

def BOIM_fourier_hodgelaplacians_combined_0_1_max_dim_1_M_M(G, simplices, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    # test max dimension = 1
    hodge = HodgeLaplacians(simplices, maxdimension=1)  # or higher, depending on the data

    L_0 = hodge.getHodgeLaplacian(d=0)  # Using 0-simplices (nodes)
    # print(hodge_laplacian)  
    hodge = HodgeLaplacians(simplices, maxdimension=2) 
    L_1 = hodge.getHodgeLaplacian(d=1)

    # Get the incidence matrix
    B = nx.incidence_matrix(G, oriented=True)  # Set 'oriented=True' for a directed incidence matrix

    m_m_matrix = get_m_m_from_hodge(L_0, L_1, B)

    # Compute eigenvalues and eigenvectors of the Hodge Laplacian
    _, eig_vect = np.linalg.eigh(m_m_matrix.toarray())  # Make sure to convert to dense if necessary
    UT = np.linalg.inv(eig_vect)  # Inverse of eigenvectors
    UT_inv = eig_vect  # Directly use eigenvectors

    shape = UT.shape
    # print("Shape of the UT:", shape)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    # print("candidate_sets", candidate_sets)
    # print("len(candidate_sets)", len(candidate_sets))

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets ,UT)
    # print("sets_after_fourier_transfer", sets_after_fourier_transfer)
    # print("len(sets_after_fourier_transfer)", len(sets_after_fourier_transfer))

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_

    groups = [[] for i in range(number_of_clusters)]

    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])

    train_X = []
    train_Y = []

    for i in range(number_of_clusters):

        selected_signal = random.choice(groups[i])
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)
        # print("source_set", source_set)
        # print("len(source_set)", len(source_set))

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(number_of_clusters):

            samples = random.sample(groups[i], 1)
            for sample in samples:
                inputs.append(torch.FloatTensor(sample))

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best = float('-inf')
    identified_signal = None

    for signal in sets_after_fourier_transfer:

        input = torch.FloatTensor([signal])
        y_pred = model(input).loc
        if y_pred > best:
            best = y_pred
            identified_signal = signal

    identified_set = find_source_set_from_fourier(identified_signal, number_of_sources, UT_inv)

    return identified_set

def BOIM_fourier_hodgelaplacians_combined_0_1_max_dim_2_M_M(G, simplices, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    # test max dimension = 2
    hodge = HodgeLaplacians(simplices, maxdimension=2)  # or higher, depending on the data

    L_0 = hodge.getHodgeLaplacian(d=0)  # Using 0-simplices (nodes)
    # print(hodge_laplacian)  
    hodge = HodgeLaplacians(simplices, maxdimension=2) 
    L_1 = hodge.getHodgeLaplacian(d=1)

    # Get the incidence matrix
    B = nx.incidence_matrix(G, oriented=True)  # Set 'oriented=True' for a directed incidence matrix

    m_m_matrix = get_m_m_from_hodge(L_0, L_1, B)

    # Compute eigenvalues and eigenvectors of the Hodge Laplacian
    _, eig_vect = np.linalg.eigh(m_m_matrix.toarray())  # Make sure to convert to dense if necessary
    UT = np.linalg.inv(eig_vect)  # Inverse of eigenvectors
    UT_inv = eig_vect  # Directly use eigenvectors

    shape = UT.shape
    # print("Shape of the UT:", shape)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    # print("candidate_sets", candidate_sets)
    # print("len(candidate_sets)", len(candidate_sets))

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets ,UT)
    # print("sets_after_fourier_transfer", sets_after_fourier_transfer)
    # print("len(sets_after_fourier_transfer)", len(sets_after_fourier_transfer))

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_

    groups = [[] for i in range(number_of_clusters)]

    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])

    train_X = []
    train_Y = []

    for i in range(number_of_clusters):

        selected_signal = random.choice(groups[i])
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)
        # print("source_set", source_set)
        # print("len(source_set)", len(source_set))

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(number_of_clusters):

            samples = random.sample(groups[i], 1)
            for sample in samples:
                inputs.append(torch.FloatTensor(sample))

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best = float('-inf')
    identified_signal = None

    for signal in sets_after_fourier_transfer:

        input = torch.FloatTensor([signal])
        y_pred = model(input).loc
        if y_pred > best:
            best = y_pred
            identified_signal = signal

    identified_set = find_source_set_from_fourier(identified_signal, number_of_sources, UT_inv)

    return identified_set

def BOIM_fourier_hodgelaplacians_0_1_combined_padding(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):

    L = compute_padding(G, alpha=0.1)
    compute_eigen_decomposition
    UT, UT_inv = compute_eigen_decomposition(L)

    shape = UT.shape
    # print("Shape of the UT:", shape)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets, UT)
    
    groups = perform_clustering_and_select_signals(number_of_clusters, sets_after_fourier_transfer)

    train_X, train_Y = simulate_initial_training(groups, G, config, num_of_sims, diffusion_model, number_of_sources, UT_inv)

    train_X, train_Y, function_values, acquisition_values, max_train_Y_values, model = iterative_optimization(train_X, train_Y, groups, number_of_sources, UT_inv, G, config, num_of_sims, diffusion_model, num_iterations)

    identified_set = identify_best_signal(sets_after_fourier_transfer, model, number_of_sources, UT_inv)
    
    return identified_set

def BOIM_fourier_hodgelaplacians_0_1_combined_heat_kernel(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    # Compute the combined heat kernel and its eigendecomposition
    L = combined_heat_kernel(G, t=1, alpha=0.5)
    UT, UT_inv = compute_eigen_decomposition(L)
    print("shape of UT", UT.shape)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets, UT)
    
    groups = perform_clustering_and_select_signals(number_of_clusters, sets_after_fourier_transfer)

    train_X, train_Y = simulate_initial_training(groups, G, config, num_of_sims, diffusion_model, number_of_sources, UT_inv)

    train_X, train_Y, function_values, acquisition_values, max_train_Y_values, model = iterative_optimization(train_X, train_Y, groups, number_of_sources, UT_inv, G, config, num_of_sims, diffusion_model, num_iterations)

    identified_set = identify_best_signal(sets_after_fourier_transfer, model, number_of_sources, UT_inv)
    
    return identified_set


################################ hodgelaplacians normalized
# zijian 
def BOIM_fourier_hodgelaplacians_0_max_dim_1_nl_sparse(G, simplices, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    # test max dimension = 1
    hodge = HodgeLaplacians(simplices, maxdimension=1)  # or higher, depending on the data
    hodge_laplacian = hodge.getHodgeLaplacian(d=0)  # Using 0-simplices (nodes)
   
    nl_sparse = normalize_laplacian_sparse(hodge_laplacian)

    # Compute eigenvalues and eigenvectors of the Hodge Laplacian
    _, eig_vect = np.linalg.eigh(nl_sparse.toarray())  # Make sure to convert to dense if necessary
    UT = np.linalg.inv(eig_vect)  # Inverse of eigenvectors
    UT_inv = eig_vect  # Directly use eigenvectors

    shape = UT.shape
    # print("Shape of the UT:", shape)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    # print("candidate_sets", candidate_sets)
    # print("len(candidate_sets)", len(candidate_sets))

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets ,UT)
    # print("sets_after_fourier_transfer", sets_after_fourier_transfer)
    # print("len(sets_after_fourier_transfer)", len(sets_after_fourier_transfer))

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_

    groups = [[] for i in range(number_of_clusters)]

    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])

    train_X = []
    train_Y = []

    for i in range(number_of_clusters):

        selected_signal = random.choice(groups[i])
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)
        # print("source_set", source_set)
        # print("len(source_set)", len(source_set))

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(number_of_clusters):

            samples = random.sample(groups[i], 1)
            for sample in samples:
                inputs.append(torch.FloatTensor(sample))

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best = float('-inf')
    identified_signal = None

    for signal in sets_after_fourier_transfer:

        input = torch.FloatTensor([signal])
        y_pred = model(input).loc
        if y_pred > best:
            best = y_pred
            identified_signal = signal

    identified_set = find_source_set_from_fourier(identified_signal, number_of_sources, UT_inv)

    return identified_set

def BOIM_fourier_hodgelaplacians_0_max_dim_2_nl_sparse(G, simplices, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):

    hodge = HodgeLaplacians(simplices, maxdimension=2)  # or higher, depending on the data
    hodge_laplacian = hodge.getHodgeLaplacian(d=0)  # Using 0-simplices (nodes)
    # print(hodge_laplacian)  

    nl_sparse = normalize_laplacian_sparse(hodge_laplacian)

    # Compute eigenvalues and eigenvectors of the Hodge Laplacian
    _, eig_vect = np.linalg.eigh(nl_sparse.toarray())
    UT = np.linalg.inv(eig_vect)  # Inverse of eigenvectors
    UT_inv = eig_vect  # Directly use eigenvectors

    shape = UT.shape
    # print("Shape of the UT:", shape)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    # print("candidate_sets", candidate_sets)
    # print("len(candidate_sets)", len(candidate_sets))

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets ,UT)
    # print("sets_after_fourier_transfer", sets_after_fourier_transfer)
    # print("len(sets_after_fourier_transfer)", len(sets_after_fourier_transfer))

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_

    groups = [[] for i in range(number_of_clusters)]

    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])

    train_X = []
    train_Y = []

    for i in range(number_of_clusters):

        selected_signal = random.choice(groups[i])
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)
        # print("source_set", source_set)
        # print("len(source_set)", len(source_set))

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(number_of_clusters):

            samples = random.sample(groups[i], 1)
            for sample in samples:
                inputs.append(torch.FloatTensor(sample))

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best = float('-inf')
    identified_signal = None

    for signal in sets_after_fourier_transfer:

        input = torch.FloatTensor([signal])
        y_pred = model(input).loc
        if y_pred > best:
            best = y_pred
            identified_signal = signal

    identified_set = find_source_set_from_fourier(identified_signal, number_of_sources, UT_inv)

    return identified_set

def BOIM_fourier_hodgelaplacians_1_nl_sparse(G, simplices, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    hodge = HodgeLaplacians(simplices, maxdimension=2)  # or higher, depending on the data
    hodge_laplacian = hodge.getHodgeLaplacian(d=1)  # Using 1-simplices (edges)
    # print(hodge_laplacian)  

    nl_sparse = normalize_laplacian_sparse(hodge_laplacian)

    # Compute eigenvalues and eigenvectors of the Hodge Laplacian
    _, eig_vect = np.linalg.eigh(nl_sparse.toarray())

    UT = np.linalg.inv(eig_vect)  # Inverse of eigenvectors
    UT_inv = eig_vect  # Directly use eigenvectors

    shape = UT.shape
    # print("Shape of the UT:", shape)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    # candidate_sets = create_candidate_set_pool_higher_degree(simplices, candidate_size=100, degree=1)
    # print("candidate_sets", candidate_sets)
    # print("len(candidate_sets)", len(candidate_sets))

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets ,UT)
    # print("sets_after_fourier_transfer", sets_after_fourier_transfer)
    # print("len(sets_after_fourier_transfer)", len(sets_after_fourier_transfer))

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_

    groups = [[] for i in range(number_of_clusters)]

    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])

    train_X = []
    train_Y = []

    for i in range(number_of_clusters):

        selected_signal = random.choice(groups[i])
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)
        # print("source_set", source_set)
        # print("len(source_set)", len(source_set))

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(number_of_clusters):

            samples = random.sample(groups[i], 1)
            for sample in samples:
                inputs.append(torch.FloatTensor(sample))

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best = float('-inf')
    identified_signal = None

    for signal in sets_after_fourier_transfer:

        input = torch.FloatTensor([signal])
        y_pred = model(input).loc
        if y_pred > best:
            best = y_pred
            identified_signal = signal

    identified_set = find_source_set_from_fourier(identified_signal, number_of_sources, UT_inv)

    return identified_set

def BOIM_fourier_hodgelaplacians_0_1_combined_padding_nl_sparse(G, simplices, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):

    hodge = HodgeLaplacians(simplices, maxdimension=2)  # or higher, depending on the data
    # print(hodge_laplacian)  
    alpha=0.5

    L_0 = hodge.getHodgeLaplacian(d=0).toarray()
    L_1 = hodge.getHodgeLaplacian(d=1).toarray()

    max_dim = max(L_0.shape[0], L_1.shape[0])
    padded_L_0 = np.zeros((max_dim, max_dim))
    padded_L_1 = np.zeros((max_dim, max_dim))

    padded_L_0[:L_0.shape[0], :L_0.shape[1]] = L_0
    padded_L_1[:L_1.shape[0], :L_1.shape[1]] = L_1

    # Weighted combination of padded Hodge Laplacians
    L_combined = alpha * padded_L_0 + (1 - alpha) * padded_L_1

    nl_sparse = normalize_laplacian_sparse(L_combined)

    # Compute eigenvalues and eigenvectors of the Hodge Laplacian
    _, eig_vect = np.linalg.eigh(nl_sparse)  # Make sure to convert to dense if necessary
    UT = np.linalg.inv(eig_vect)  # Inverse of eigenvectors
    UT_inv = eig_vect  # Directly use eigenvectors

    shape = UT.shape
    # print("Shape of the UT:", shape)

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    # print("candidate_sets", candidate_sets)
    # print("len(candidate_sets)", len(candidate_sets))

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets ,UT)
    # print("sets_after_fourier_transfer", sets_after_fourier_transfer)
    # print("len(sets_after_fourier_transfer)", len(sets_after_fourier_transfer))

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_

    groups = [[] for i in range(number_of_clusters)]

    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])

    train_X = []
    train_Y = []

    for i in range(number_of_clusters):

        selected_signal = random.choice(groups[i])
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)
        # print("source_set", source_set)
        # print("len(source_set)", len(source_set))

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(number_of_clusters):

            samples = random.sample(groups[i], 1)
            for sample in samples:
                inputs.append(torch.FloatTensor(sample))

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best = float('-inf')
    identified_signal = None

    for signal in sets_after_fourier_transfer:

        input = torch.FloatTensor([signal])
        y_pred = model(input).loc
        if y_pred > best:
            best = y_pred
            identified_signal = signal

    identified_set = find_source_set_from_fourier(identified_signal, number_of_sources, UT_inv)

    return identified_set

################################ hodge laplacians Todos
# FIXME
def BOIM_fourier_hodgelaplacians_0_1_combined_mapping(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    
    simplices = generate_simplices(G, dimension=1)
    hodge = HodgeLaplacians(simplices, maxdimension=1)

    L_0 = hodge.getHodgeLaplacian(d=0).toarray()
    L_1 = hodge.getHodgeLaplacian(d=1).toarray()

    UT_node, UT_node_inv = compute_eigen_decomposition(L_0)
    UT_edge, UT_edge_inv = compute_eigen_decomposition(L_1)

    node_candidates = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    edge_candidates = create_candidate_set_pool_filtering_mapping(G, candidate_size, number_of_sources, allowed_shortest_distance)

    node_groups, node_signals = perform_clustering_and_select_signals(number_of_clusters, node_candidates)
    edge_groups, edge_signals = perform_clustering_and_select_signals(number_of_clusters, edge_candidates)

    train_X_nodes, train_Y_nodes = simulate_initial_training(node_groups, G, config, num_of_sims, diffusion_model, number_of_sources, UT_node_inv)
    train_X_edges, train_Y_edges = simulate_initial_training(edge_groups, G, config, num_of_sims, diffusion_model, number_of_sources, UT_edge_inv)

    # Determine the number of components as the minimum dimension of node or edge features that can be used
    n_components = min(train_X_nodes.shape[1], train_X_edges.shape[1], train_X_nodes.shape[0], train_X_edges.shape[0])

    train_X_nodes = reduce_features(train_X_nodes, n_components)
    train_X_edges = reduce_features(train_X_edges, n_components)

    train_X = np.concatenate((train_X_nodes, train_X_edges), axis=0)
    train_Y = np.concatenate((train_Y_nodes, train_Y_edges), axis=0)

    train_X, train_Y, function_values, acquisition_values, max_train_Y_values, model = iterative_optimization(train_X, train_Y, groups, number_of_sources, UT_inv, G, config, num_of_sims, diffusion_model, num_iterations)

    signal = node_signals + edge_signals
    identified_set = identify_best_signal(signal, model, number_of_sources, UT_node_inv, UT_edge_inv)

    return identified_set

def BOIM_fourier_hodgelaplacians_0_1_combined_heat_kernel_nl_sparse(G, simplices, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):
    # Compute the combined heat kernel and its eigendecomposition
    combined_kernel = combined_heat_kernel_nl_sparse(G, simplices, t=1.0, alpha=0.5)
    _, eig_vect = np.linalg.eigh(combined_kernel)  # Eigen decomposition
    UT = np.linalg.inv(eig_vect)  # Inverse of eigenvectors
    UT_inv = eig_vect  # Directly use eigenvectors

    # Other steps remain similar
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets ,UT)

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_
    groups = [[] for i in range(number_of_clusters)]
    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])

    train_X = []
    train_Y = []
    for i in range(number_of_clusters):
        selected_signal = random.choice(groups[i])
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)
        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
        
        input = torch.FloatTensor(selected_signal)
        train_X.append(input)
        train_Y.append([float(e)])
    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)


    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(number_of_clusters):

            samples = random.sample(groups[i], 1)
            for sample in samples:
                inputs.append(torch.FloatTensor(sample))

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best = float('-inf')
    identified_signal = None

    for signal in sets_after_fourier_transfer:

        input = torch.FloatTensor([signal])
        y_pred = model(input).loc
        if y_pred > best:
            best = y_pred
            identified_signal = signal

    identified_set = find_source_set_from_fourier(identified_signal, number_of_sources, UT_inv)

    return identified_set

def BOIM_fourier_hodgelaplacians_2(G, simplices, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):

    hodge = HodgeLaplacians(simplices, maxdimension=2)  # or higher, depending on the data
    hodge_laplacian = hodge.getHodgeLaplacian(d=2)  # Using 1-simplices (edges)
    # print(hodge_laplacian)  

    # Compute eigenvalues and eigenvectors of the Hodge Laplacian
    _, eig_vect = np.linalg.eigh(hodge_laplacian.toarray())  # Make sure to convert to dense if necessary
    UT = np.linalg.inv(eig_vect)  # Inverse of 
    UT_inv = eig_vect  # Directly use eigenvectors

    shape = UT.shape
    # print("UT:", UT)
    # print("Shape of the UT:", shape)
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    # print("len(candidate_sets)", len(candidate_sets))
    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets ,UT)
    # print("len(sets_after_fourier_transfer)", len(sets_after_fourier_transfer))

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_

    groups = [[] for i in range(number_of_clusters)]

    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])

    train_X = []
    train_Y = []

    for i in range(number_of_clusters):

        selected_signal = random.choice(groups[i])
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)
        # print("len(source_set)", len(source_set))

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(number_of_clusters):

            samples = random.sample(groups[i], 1)
            for sample in samples:
                inputs.append(torch.FloatTensor(sample))

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best = float('-inf')
    identified_signal = None

    for signal in sets_after_fourier_transfer:

        input = torch.FloatTensor([signal])
        y_pred = model(input).loc
        if y_pred > best:
            best = y_pred
            identified_signal = signal

    identified_set = find_source_set_from_fourier(identified_signal, number_of_sources, UT_inv)

    return identified_set

def BOIM_fourier_graph_to_hodge_0(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):

    # nl = nx.normalized_laplacian_matrix(G)
    print("graph to hodge 0")
    L_1 = get_laplacian_0(G)
    _, eig_vect = np.linalg.eigh(L_1.toarray())
    UT = np.linalg.inv(eig_vect)
    UT_inv = eig_vect

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets ,UT)

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_

    groups = [[] for i in range(number_of_clusters)]

    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])

    train_X = []
    train_Y = []

    for i in range(number_of_clusters):

        selected_signal = random.choice(groups[i])
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(number_of_clusters):

            samples = random.sample(groups[i], 1)
            for sample in samples:
                inputs.append(torch.FloatTensor(sample))

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best = float('-inf')
    identified_signal = None

    for signal in sets_after_fourier_transfer:

        input = torch.FloatTensor([signal])
        y_pred = model(input).loc
        if y_pred > best:
            best = y_pred
            identified_signal = signal

    identified_set = find_source_set_from_fourier(identified_signal, number_of_sources, UT_inv)

    return identified_set

def BOIM_fourier_graph_to_hodge_1(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):

    # nl = nx.normalized_laplacian_matrix(G)
    print("graph to hodge 1")
    L_1 = get_laplacian_1(G)
    _, eig_vect = np.linalg.eigh(L_1.toarray())
    UT = np.linalg.inv(eig_vect)
    UT_inv = eig_vect

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets ,UT)

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_

    groups = [[] for i in range(number_of_clusters)]

    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])

    train_X = []
    train_Y = []

    for i in range(number_of_clusters):

        selected_signal = random.choice(groups[i])
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(number_of_clusters):

            samples = random.sample(groups[i], 1)
            for sample in samples:
                inputs.append(torch.FloatTensor(sample))

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best = float('-inf')
    identified_signal = None

    for signal in sets_after_fourier_transfer:

        input = torch.FloatTensor([signal])
        y_pred = model(input).loc
        if y_pred > best:
            best = y_pred
            identified_signal = signal

    identified_set = find_source_set_from_fourier(identified_signal, number_of_sources, UT_inv)

    return identified_set

def BOIM_fourier_graph_to_hodge_2(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):

    # nl = nx.normalized_laplacian_matrix(G)
    print("graph to hodge 1")
    L_2 = get_laplacian_2(G)
    _, eig_vect = np.linalg.eigh(L_2.toarray())
    UT = np.linalg.inv(eig_vect)
    UT_inv = eig_vect

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets ,UT)

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_

    groups = [[] for i in range(number_of_clusters)]

    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])

    train_X = []
    train_Y = []

    for i in range(number_of_clusters):

        selected_signal = random.choice(groups[i])
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(number_of_clusters):

            samples = random.sample(groups[i], 1)
            for sample in samples:
                inputs.append(torch.FloatTensor(sample))

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best = float('-inf')
    identified_signal = None

    for signal in sets_after_fourier_transfer:

        input = torch.FloatTensor([signal])
        y_pred = model(input).loc
        if y_pred > best:
            best = y_pred
            identified_signal = signal

    identified_set = find_source_set_from_fourier(identified_signal, number_of_sources, UT_inv)

    return identified_set

def BOIM_fourier_graph_to_hodge_3(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance, number_of_clusters):

    # nl = nx.normalized_laplacian_matrix(G)
    L_3 = get_laplacian_3(G)
    _, eig_vect = np.linalg.eigh(L_3.toarray())
    UT = np.linalg.inv(eig_vect)
    UT_inv = eig_vect

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets ,UT)

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_

    groups = [[] for i in range(number_of_clusters)]

    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])

    train_X = []
    train_Y = []

    for i in range(number_of_clusters):

        selected_signal = random.choice(groups[i])
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(number_of_clusters):

            samples = random.sample(groups[i], 1)
            for sample in samples:
                inputs.append(torch.FloatTensor(sample))

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best = float('-inf')
    identified_signal = None

    for signal in sets_after_fourier_transfer:

        input = torch.FloatTensor([signal])
        y_pred = model(input).loc
        if y_pred > best:
            best = y_pred
            identified_signal = signal

    identified_set = find_source_set_from_fourier(identified_signal, number_of_sources, UT_inv)

    return identified_set

# zijian TODO
def BOIM_fourier_no_GSS_hodgelaplacians(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance):

    # nl = nx.normalized_laplacian_matrix(G)
    # _, eig_vect = np.linalg.eigh(nl.todense())
    # UT = np.linalg.inv(eig_vect)
    # UT_inv = eig_vect

    simplices = []
    for node in G.nodes:
        for neighbor in G[node]:
            simplices.append((node, neighbor))  # 1-simplex (edge)
            for second_neighbor in G[neighbor]:
                if G.has_edge(node, second_neighbor) and node != second_neighbor:
                    simplices.append(tuple(sorted((node, neighbor, second_neighbor))))  # 2-simplex (triangle)
    
    print('number of simplices:', len(simplices))

    # Assuming `simplices` = already defined and filled with data from graph G
    hodge = HodgeLaplacians(simplices, maxdimension=1)  # or higher, depending on the data
    hodge_laplacian = hodge.getHodgeLaplacian(d=1)  # Using 1-simplices (edges)
    # print(hodge_laplacian)  

    # Compute eigenvalues and eigenvectors of the Hodge Laplacian
    _, eig_vect = np.linalg.eigh(hodge_laplacian.toarray())  # Make sure to convert to dense if necessary
    UT = np.linalg.inv(eig_vect)  # Inverse of eigenvectors
    UT_inv = eig_vect  # Directly use eigenvectors

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)

    train_X = []
    train_Y = []

    for i in range(num_initial_samples):

        selected_set = random.sample(candidate_sets, 1)[0]
        selected_signal = create_signal_from_source_set(G, selected_set, UT)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
        
        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(num_initial_samples):

            selected_set = random.sample(candidate_sets, 1)[0]
            selected_signal = create_signal_from_source_set(G, selected_set, UT)

            input = torch.FloatTensor(selected_signal)
            inputs.append(input)

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    # final_index = sets_after_fourier_transfer.index(s)
    identified_set = find_source_set_from_fourier(s, number_of_sources, UT_inv)

    return identified_set

# zijian TODO
def BOIM_no_filtering_hodgelaplacians(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):

    # nl = nx.normalized_laplacian_matrix(G)
    # _, eig_vect = np.linalg.eigh(nl.todense())
    # UT = np.linalg.inv(eig_vect)
    # UT_inv = eig_vect

    simplices = []
    for node in G.nodes:
        for neighbor in G[node]:
            simplices.append((node, neighbor))  # 1-simplex (edge)
            for second_neighbor in G[neighbor]:
                if G.has_edge(node, second_neighbor) and node != second_neighbor:
                    simplices.append(tuple(sorted((node, neighbor, second_neighbor))))  # 2-simplex (triangle)
    
    print('number of simplices:', len(simplices))

    # Assuming `simplices` = already defined and filled with data from graph G
    hodge = HodgeLaplacians(simplices, maxdimension=1)  # or higher, depending on the data
    hodge_laplacian = hodge.getHodgeLaplacian(d=1)  # Using 1-simplices (edges)
    # print(hodge_laplacian)  

    # Compute eigenvalues and eigenvectors of the Hodge Laplacian
    _, eig_vect = np.linalg.eigh(hodge_laplacian.toarray())  # Make sure to convert to dense if necessary
    UT = np.linalg.inv(eig_vect)  # Inverse of eigenvectors
    UT_inv = eig_vect  # Directly use eigenvectors

    candidate_sets = create_candidate_set_pool(G, candidate_size, number_of_sources)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets ,UT)

    train_X = []
    train_Y = []

    for i in range(num_initial_samples):

        selected_set = random.sample(candidate_sets, 1)[0]
        selected_signal = create_signal_from_source_set(G, selected_set, UT)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected_set, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []

        for i in range(num_initial_samples):

            selected_set = random.sample(candidate_sets, 1)[0]
            selected_signal = create_signal_from_source_set(G, selected_set, UT)

            input = torch.FloatTensor(selected_signal)
            inputs.append(input)

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        if diffusion_model == 'ic':
            e,_ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e,_ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([float(e)])

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    # final_index = sets_after_fourier_transfer.index(s)
    identified_set = find_source_set_from_fourier(s, number_of_sources, UT_inv)

    return identified_set

############################################

def eigen(g, config, budget):

    g_eig = g.__class__()
    g_eig.add_nodes_from(g)
    g_eig.add_edges_from(g.edges)
    for a, b in g_eig.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_eig[a][b]['weight'] = weight

    eig = []

    for k in range(budget):

        eigen = nx.eigenvector_centrality_numpy(g_eig)
        selected = sorted(eigen, key=eigen.get, reverse=True)[0]
        eig.append(selected)
        g_eig.remove_node(selected)

    return eig

def degree(g, config, budget):
    g_deg = g.__class__()
    g_deg.add_nodes_from(g)
    g_deg.add_edges_from(g.edges)
    for a, b in g_deg.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_deg[a][b]['weight'] = weight

    deg = []

    for k in range(budget):
        degree = nx.centrality.degree_centrality(g_deg)
        selected = sorted(degree, key=degree.get, reverse=True)[0]
        deg.append(selected)
        g_deg.remove_node(selected)

    return deg

def pi(g, config, budget):
    g_greedy = g.__class__()
    g_greedy.add_nodes_from(g)
    g_greedy.add_edges_from(g.edges)

    for a, b in g_greedy.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_greedy[a][b]['weight'] = weight

    result = []

    for k in range(budget):

        n = g_greedy.number_of_nodes()

        I = np.ones((n, 1))

        C = np.ones((n, n))
        N = np.ones((n, n))

        A = nx.to_numpy_array(g_greedy, nodelist=list(g_greedy.nodes()))

        for i in range(5):
            B = np.power(A, i + 1)
            D = C - B
            N = np.multiply(N, D)

        P = C - N

        pi = np.matmul(P, I)

        value = {}

        for i in range(n):
            value[list(g_greedy.nodes())[i]] = pi[i, 0]

        selected = sorted(value, key=value.get, reverse=True)[0]

        result.append(selected)

        g_greedy.remove_node(selected)

    return result

def sigma(g, config, budget):
    g_greedy = g.__class__()
    g_greedy.add_nodes_from(g)
    g_greedy.add_edges_from(g.edges)

    for a, b in g_greedy.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_greedy[a][b]['weight'] = weight

    result = []

    for k in range(budget):

        n = g_greedy.number_of_nodes()

        I = np.ones((n, 1))

        F = np.ones((n, n))
        N = np.ones((n, n))

        A = nx.to_numpy_array(g, nodelist=g_greedy.nodes())

        sigma = I
        for i in range(5):
            B = np.power(A, i + 1)
            C = np.matmul(B, I)
            sigma += C

        value = {}

        for i in range(n):
            value[list(g_greedy.nodes())[i]] = sigma[i, 0]

        selected = sorted(value, key=value.get, reverse=True)[0]

        result.append(selected)

        g_greedy.remove_node(selected)

    return result

def Netshield(g, config, budget):
    g_greedy = g.__class__()
    g_greedy.add_nodes_from(g)
    g_greedy.add_edges_from(g.edges)

    for a, b in g_greedy.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_greedy[a][b]['weight'] = weight

    A = nx.adjacency_matrix(g_greedy)

    lam, u = np.linalg.eigh(A.toarray())
    lam = list(lam)
    lam = lam[-1]

    u = u[:, -1]
    u = np.abs(np.real(u).flatten())
    v = (2 * lam * np.ones(len(u))) * np.power(u, 2)

    nodes = []
    for i in range(budget):
        if nodes:
            B = A[:, nodes].toarray()
            b = np.dot(B, u[nodes])
        else:
            b = np.zeros_like(u)

        score = v - 2 * b * u
        score[nodes] = -1

        nodes.append(np.argmax(score))

    return nodes

def Soboldeg(g, config, budget):
    g_deg = g.__class__()
    g_deg.add_nodes_from(g)
    g_deg.add_edges_from(g.edges)
    for a, b in g_deg.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_deg[a][b]['weight'] = weight

    deg = []

    for k in range(2*budget):
        degree = nx.centrality.degree_centrality(g_deg)
        selected = sorted(degree, key=degree.get, reverse=True)[0]
        deg.append(selected)
        g_deg.remove_node(selected)


    for j in range(budget):
        df = simulationIC(1, g, deg, config)
        ST = SobolT(df, deg)
        rank = []
        for node in sorted(ST, key=ST.get, reverse=True):
            rank.append(node)
        rem = rank.pop()
        deg.remove((rem))

    return deg

def degreeDis(g, config, budget):

    selected = []
    d = {}
    t = {}
    dd = hd.heapdict()

    for node in g.nodes():
        d[node] = sum([g[node][v]['weight'] for v in g[node]])
        dd[node] = -d[node]
        t[node] = 0

    for i in range(budget):
        seed, _ = dd.popitem()
        selected.append(seed)
        for v in g.neighbors(seed):
            if v not in selected:
                t[v] += g[seed][v]['weight']
                discount = d[v] - 2*t[v] - (d[v] - t[v])*t[v]
                dd[v] = -discount

    return selected

def SoboldegreeDis(g, config, budget):

    selected = []
    d = {}
    t = {}
    dd = hd.heapdict()

    for node in g.nodes():
        d[node] = sum([g[node][v]['weight'] for v in g[node]])
        dd[node] = -d[node]
        t[node] = 0

    for i in range(2*budget):
        seed, _ = dd.popitem()
        selected.append(seed)
        for v in g.neighbors(seed):
            if v not in selected:
                t[v] += g[seed][v]['weight']
                discount = d[v] - 2*t[v] - (d[v] - t[v])*t[v]
                dd[v] = -discount

    for j in range(budget):
        df = simulationIC(10, g, selected, config)
        ST = SobolT(df, selected)
        rank = []
        for node in sorted(ST, key=ST.get, reverse=True):
            rank.append(node)
        rem = rank.pop()
        selected.remove((rem))

    return selected


# greedy

def greedy(g, config, budget, rounds=100, model='SI', beta=0.1):
    print("model: ", model)
    model = model.upper()

    selected = []
    candidates = list(g.nodes())

    for i in range(budget):
        max_spread = 0
        index = -1
        for node in candidates:
            seed = selected + [node]

            if model == "IC":
                result = IC(g, config, seed, rounds)
            elif model == "LT":
                result = LT(g, config, seed, rounds)
            elif model == "SI":
                result = SI(g, config, seed, rounds, beta=beta)
            else:
                raise ValueError(f"Unknown model: {model}")

            mean_result = s.mean(result)
            if mean_result > max_spread:
                max_spread = mean_result
                index = node

        if index == -1:
            raise ValueError("No valid node found to select. Check the model implementation and input graph.")

        selected.append(index)
        candidates.remove(index)

    print(selected)
    return selected


def celf(g, config, budget, rounds=100, model='SI', beta=0.1): 
    print("model: ", model)
    model = model.upper()

    # Find the first node with greedy algorithm
    
    # Compute marginal gain for each node
    candidates = list(g.nodes())
    #, start_time = list(g.nodes()), time.time()
    # step 1, call a diffusion function, get the result of list
    # step 2, calculate the margin gain 
    if (model == "IC"):
        marg_gain = [s.mean(IC(g, config, [node])) for node in candidates]
    elif (model == "LT"):
        marg_gain = [s.mean(LT(g, config, [node])) for node in candidates]
    elif (model == "SI"):
         marg_gain = [s.mean(SI(g, config, [node], beta)) for node in candidates]
    # Create the sorted list of nodes and their marginal gain 
    Q = sorted(zip(candidates,marg_gain), key = lambda x: x[1],reverse=True)

    # Select the first node and remove from candidate list
    selected, spread, Q = [Q[0][0]], Q[0][1], Q[1:]
    
    # Find the next budget-1 nodes using the CELF list-sorting procedure
    
    for _ in range(budget-1):    

        check = False      
        while not check:
            
            # Recalculate spread of top node
            current = Q[0][0]
            
            # Evaluate the spread function and store the marginal gain in the list
            if (model == "IC"):
                Q[0] = (current, s.mean(IC(g, config, selected+[current]), rounds) - spread)
            elif (model == "LT"):
                Q[0] = (current, s.mean(LT(g, config, selected+[current]), rounds) - spread)
            elif (model == "SI"):
                Q[0] = (current, s.mean(SI(g, config, selected+[current]), rounds, beta) - spread)

            # Re-sort the list
            Q = sorted(Q, key = lambda x: x[1], reverse=True)

            # Check if previous top node stayed on top after the sort
            check = Q[0][0] == current

        # Select the next node
        selected.append(Q[0][0])
        spread = Q[0][1]
        
        # Remove the selected node from the list
        Q = Q[1:]

    print(selected)
    return(selected)
    # return(sorted(S),timelapse)

def celfpp(g, config, budget, rounds=100, model='SI', beta=0.1):
    model = model.upper()

    # Compute marginal gain for each node
    candidates = list(g.nodes())
    if (model == "IC"):
        marg_gain = [s.mean(IC(g, config, [node], rounds)) for node in candidates]
    elif (model == "LT"):
        marg_gain = [s.mean(LT(g, config, [node], rounds)) for node in candidates]
    elif (model == "SI"):
        marg_gain = [s.mean(SI(g, config, [node], rounds, beta)) for node in candidates]

    # Create the sorted list of nodes and their marginal gain 
    Q = sorted(zip(candidates, marg_gain), key = lambda x: x[1], reverse=True)

    # Select the first node and remove from candidate list
    selected, spread, Q = [Q[0][0]], Q[0][1], Q[1:]

    # Initialize last_seed as the first selected node
    last_seed = selected[0]
    
    # Find the next budget-1 nodes using the CELF++ procedure
    for _ in range(budget - 1):    
        check = False
        while not check:
            # Get current node and its previous computed marginal gain
            current, old_gain = Q[0][0], Q[0][1]

            # Check if the last added seed has changed
            if current != last_seed:
                # Compute new marginal gain
                if (model == "IC"):
                    new_gain = s.mean(IC(g, config, selected+[current], rounds)) - spread
                elif (model == "LT"):
                    new_gain = s.mean(LT(g, config, selected+[current], rounds)) - spread
                elif (model == "SI"):
                    new_gain = s.mean(SI(g, config, selected+[current], rounds, beta)) - spread
            else:
                # If the last added seed hasn't changed, the marginal gain remains the same
                new_gain = old_gain

            # Update the marginal gain of the current node
            Q[0] = (current, new_gain)

            # Re-sort the list
            Q = sorted(Q, key = lambda x: x[1], reverse=True)

            # Check if previous top node stayed on top after the sort
            check = Q[0][0] == current

        # Select the next node
        selected.append(Q[0][0])
        spread += Q[0][1]  # Update the spread
        last_seed = Q[0][0]  # Update the last added seed

        # Remove the selected node from the list
        Q = Q[1:]

    print(selected)
    return selected

####################################

# IMRank
# https://github.com/Braylon1002/IMTool
def IMRank(g, config, budget):
    """
    IMRank algorithm to rank the nodes based on their influence.
    """

    # Obtain adjacency matrix from the graph
    adjacency_matrix = nx.adjacency_matrix(g).todense()

    # Normalize the adjacency matrix
    row_sums = adjacency_matrix.sum(axis=1)
    
    # Check for zero entries in row_sums (which could correspond to isolated nodes)
    # and replace them with 1 to prevent division by zero errors
    row_sums[row_sums == 0] = 1

    adjacency_matrix = adjacency_matrix / row_sums
    
    start = time.perf_counter()
    t = 0
    r0 = [i for i in range(len(adjacency_matrix))]
    r = [0 for i in range(len(adjacency_matrix))]

    # Loop until the ranks converge
    while True:
        t = t + 1
        r = LFA(adjacency_matrix)
        r = np.argsort(-np.array(r))
        if operator.eq(list(r0), list(r)):
            break
        r0 = copy.copy(r)
        
    # Select top nodes up to the budget
    selected = r[:budget].tolist()

    print(selected)
    return selected

#RIS
# https://github.com/Braylon1002/IMTool
def RIS(g, config, budget, rounds=2000):
#     mc = 100
    # Generate mc RRSs
    R = [get_RRS(g, config) for _ in range(rounds)]

    selected = []
    for _ in range(budget):
        # Collect all nodes from all RRSs
        flat_map = [item for subset in R for item in subset]
        # Only proceed if there = nodes in the flat_map
        if flat_map:
            seed = Counter(flat_map).most_common()[0][0]
            selected.append(seed)

            R = [rrs for rrs in R if seed not in rrs]

            # For every removed RRS, generate a new one
            while len(R) < rounds:
                R.append(get_RRS(g, config))

    print(selected)
    return (selected)

########### IMM ###########
# https://github.com/snowgy/Influence_Maximization

def IMM(graph, config, seed_size, model):
    model = model.upper()
    l = 1
    epsoid = 0.1
    n = graph.number_of_nodes()
    k = seed_size
    l = l * (1 + math.log(2) / math.log(n))
    R = sampling_imm(epsoid, l, graph, n, seed_size, model)
    Sk, z = node_selection(R, k, n)
    return Sk

#################################### NNGP ####################################
def BOIM_vanilla_random_nngp(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    data = load_data("Cora", transform=transition_matrix())
    # data = load_data("CiteSeer", transform=transition_matrix())

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    
    candidate_nodes = [node for node, _ in deg[:candidate_size]]

    train_X = []
    train_Y = []
    source_sets = []

    initial_indices = random.sample(range(len(candidate_sets)), 23)

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = [1 if item in source_set else 0 for item in candidate_nodes]
        input = torch.tensor(input, dtype=torch.double)

        train_X.append(input)
        train_Y.append([float(e)])  # Ensure train_Y is 2-dimensional
        source_sets.append(source_set)

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y, dtype=torch.double)

    # print("train_Y", train_Y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnngp_model = GNNGP(data, L=2, sigma_b=0.751087517315013, sigma_w=1.301961883639593, device=device)
    
    epsilon = torch.logspace(-3, 1, 101, device=device)
    gnngp_model.predict(epsilon)

    wrapped_gnngp_model = GNNGPBoTorchWrapper(gnngp_model, epsilon)

    acq_func = UpperConfidenceBound(model=wrapped_gnngp_model, beta=1.5489486601510272)
    function_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        candidate_inputs = []
        for _ in range(candidate_size):
            candidate_set = random.sample(candidate_nodes, number_of_sources)
            candidate_input = [1 if item in candidate_set else 0 for item in candidate_nodes]
            candidate_inputs.append(torch.tensor(candidate_input, dtype=torch.double))

        candidate_inputs = torch.stack(candidate_inputs)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=candidate_inputs,
        )

        found_candidate = candidate[0]

        signal = found_candidate.tolist()
        selected = [candidate_nodes[i] for i in range(candidate_size) if int(signal[i]) == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)
        # print(f"New Y: {new_Y}")
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        max_train_Y_values.append(train_Y.max().item())

        try:
            gnngp_model.predict(epsilon)
        except RuntimeError as e:
            print(f"Error at iteration {iteration}:")
            print(f"train_X shape: {train_X.shape}, train_Y shape: {train_Y.shape}")
            print(f"train_X:\n{train_X}")
            print(f"train_Y:\n{train_Y}")
            print(f"Model state dict: {gnngp_model.state_dict()}")
            raise e

    best_index = train_Y.argmax().item()
    # print('best_index', best_index)
    # print('train_Y max', train_Y.max().item())
    s = list(train_X[best_index])
    result = [candidate_nodes[i] for i in range(candidate_size) if int(s[i]) == 1]
    # print('result', result)

    return result


def BOIM_vanilla_top_nngp(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    data = load_data("Cora", transform=transition_matrix())
    # data = load_data("CiteSeer", transform=transition_matrix())

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidate_nodes = [node for node, _ in deg[:candidate_size]]

    candidates = enhance_candidate_selection_with_centrality_normalized(G, candidate_nodes)

    train_X = []
    train_Y = []
    source_sets = []

    top_indices = range(3)

    initial_indices = list(top_indices)

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input, dtype=torch.double)

        train_X.append(input)
        train_Y.append([float(e)])  # Ensure train_Y is 2-dimensional
        source_sets.append(source_set)

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y, dtype=torch.double)

    print("train_Y", train_Y)
    
    # Define and train the GNNGP model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gnngp_model = GNNGP(data, L=2, sigma_b=0.0, sigma_w=1.0, device=device)
    gnngp_model = GNNGP(data, L=2, sigma_b=0.751087517315013, sigma_w=1.301961883639593, device=device)
    
    epsilon = torch.logspace(-3, 1, 101, device=device)
    # epsilon = torch.logspace(-2, 0, 101, device=device)
    gnngp_model.predict(epsilon)  # Train the model and perform prediction
    
    # Wrap GNNGP model to be compatible with BoTorch
    wrapped_gnngp_model = GNNGPBoTorchWrapper(gnngp_model, epsilon)

    # acq_func = ExpectedImprovement(model=wrapped_gnngp_model, best_f=train_Y.max().item())
    acq_func = UpperConfidenceBound(model=wrapped_gnngp_model, beta=1.5489486601510272)
    function_values = []  # Initialize function_values list
    max_train_Y_values = [train_Y.max().item()]  # Initialize max_train_Y_values list with initial max value

    for iteration in range(num_iterations):
        candidate_inputs = []
        for _ in range(candidate_size):
            candidate_set = random.sample(candidate_nodes, number_of_sources)
            candidate_input = [1 if item in candidate_set else 0 for item in candidates]
            candidate_inputs.append(torch.tensor(candidate_input, dtype=torch.double))

        candidate_inputs = torch.stack(candidate_inputs)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=candidate_inputs,
        )

        found_candidate = candidate[0]

        signal = found_candidate.tolist()
        # print(f"Signal: {signal}")
        selected = [candidates[i] for i in range(candidate_size) if int(signal[i]) == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
            # print('e', e)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)  # Ensure new_Y is 2-dimensional
        print(f"New Y: {new_Y}")
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)
        # print('train_Y 2', train_Y)

        function_values.append(new_Y.item())
        max_train_Y_values.append(train_Y.max().item())
        # print('max_train_Y_values', max_train_Y_values)

        try:
            gnngp_model.predict(epsilon)  # Update the GNNGP model
        except RuntimeError as e:
            print(f"Error at iteration {iteration}:")
            print(f"train_X shape: {train_X.shape}, train_Y shape: {train_Y.shape}")
            print(f"train_X:\n{train_X}")
            print(f"train_Y:\n{train_Y}")
            print(f"Model state dict: {gnngp_model.state_dict()}")
            raise e

    best_index = train_Y.argmax().item()
    print('best_index', best_index)
    print('train_Y max', train_Y.max().item())
    s = list(train_X[best_index])
    result = [candidates[i] for i in range(candidate_size) if int(s[i]) == 1]
    print('result', result)
    return result

def BOIM_vanilla_top_ntk(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidate_nodes = [node for node, _ in deg[:candidate_size]]

    candidates = enhance_candidate_selection_with_centrality_normalized(G, candidate_nodes)

    train_X = []
    train_Y = []
    source_sets = []

    top_indices = range(3)

    for index in top_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input, dtype=torch.double)

        train_X.append(input)
        train_Y.append([float(e)])  # Ensure train_Y is 2-dimensional
        source_sets.append(source_set)

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y, dtype=torch.double)
    # print("train_Y", train_Y)

    # Define the NTK model
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(128), stax.Relu(), stax.Dense(128), stax.Relu(), stax.Dense(train_X.size(1))  # Match last layer to input size
    )

    ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
    ntk_kernel = np.array(ntk_kernel)
    
    epsilon = torch.logspace(-3, 1, 101, device='cpu')
    wrapped_ntk_model = NTKBoTorchWrapper(ntk_kernel, epsilon)

    acq_func = ExpectedImprovement(model=wrapped_ntk_model, best_f=train_Y.max().item())

    function_values = []  # Initialize function_values list
    max_train_Y_values = [train_Y.max().item()]  # Initialize max_train_Y_values list with initial max value

    for iteration in range(num_iterations):
        candidate_inputs = []
        for _ in range(candidate_size):
            candidate_set = random.sample(candidate_nodes, number_of_sources)
            candidate_input = [1 if item in candidate_set else 0 for item in candidates]
            candidate_inputs.append(torch.tensor(candidate_input, dtype=torch.double))

        candidate_inputs = torch.stack(candidate_inputs)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=candidate_inputs,
        )

        found_candidate = candidate[0]

        signal = found_candidate.tolist()
        selected = [candidates[i] for i in range(candidate_size) if int(signal[i]) == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)  # Ensure new_Y is 2-dimensional
        # print('new_Y', new_Y)

        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        max_train_Y_values.append(train_Y.max().item())

        try:
            ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
            ntk_kernel = np.array(ntk_kernel)
            wrapped_ntk_model.ntk_kernel = torch.tensor(ntk_kernel, dtype=torch.float32)
        except RuntimeError as e:
            print(f"Error at iteration {iteration}:")
            print(f"train_X shape: {train_X.shape}, train_Y shape: {train_Y.shape}")
            print(f"train_X:\n{train_X}")
            print(f"train_Y:\n{train_Y}")
            raise e

    best_index = train_Y.argmax().item()
    # print('best_index', best_index)
    s = list(train_X[best_index])
    result = [candidates[i] for i in range(candidate_size) if int(s[i]) == 1]
    # print('result', result)
    return result

def BOIM_vanilla_random_ntk(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    
    candidate_nodes = [node for node, _ in deg[:candidate_size]]

    train_X = []
    train_Y = []
    source_sets = []

    initial_indices = random.sample(range(len(candidate_sets)), 23)

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = [1 if item in source_set else 0 for item in candidate_nodes]
        input = torch.tensor(input, dtype=torch.double)

        train_X.append(input)
        train_Y.append([float(e)])  # Ensure train_Y is 2-dimensional
        source_sets.append(source_set)

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y, dtype=torch.double)
    print("train_Y", train_Y)

    # Define the NTK model
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(128), stax.Relu(), stax.Dense(128), stax.Relu(), stax.Dense(train_X.size(1))  # Match last layer to input size
    )

    ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
    ntk_kernel = np.array(ntk_kernel)
    
    epsilon = torch.logspace(-3, 1, 101, device='cpu')
    wrapped_ntk_model = NTKBoTorchWrapper(ntk_kernel, epsilon)

    acq_func = ExpectedImprovement(model=wrapped_ntk_model, best_f=train_Y.max().item())

    function_values = []  # Initialize function_values list
    max_train_Y_values = [train_Y.max().item()]  # Initialize max_train_Y_values list with initial max value

    for iteration in range(num_iterations):
        candidate_inputs = []
        for _ in range(candidate_size):
            candidate_set = random.sample(candidate_nodes, number_of_sources)
            candidate_input = [1 if item in candidate_set else 0 for item in candidate_nodes]
            candidate_inputs.append(torch.tensor(candidate_input, dtype=torch.double))

        candidate_inputs = torch.stack(candidate_inputs)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=candidate_inputs,
        )

        found_candidate = candidate[0]

        signal = found_candidate.tolist()
        selected = [candidate_nodes[i] for i in range(candidate_size) if int(signal[i]) == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)  # Ensure new_Y is 2-dimensional
        # print('new_Y', new_Y)

        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        max_train_Y_values.append(train_Y.max().item())

        try:
            ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
            ntk_kernel = np.array(ntk_kernel)
            wrapped_ntk_model.ntk_kernel = torch.tensor(ntk_kernel, dtype=torch.float32)
        except RuntimeError as e:
            print(f"Error at iteration {iteration}:")
            print(f"train_X shape: {train_X.shape}, train_Y shape: {train_Y.shape}")
            print(f"train_X:\n{train_X}")
            print(f"train_Y:\n{train_Y}")
            raise e

    best_index = train_Y.argmax().item()
    # print('best_index', best_index)
    s = list(train_X[best_index])
    result = [candidate_nodes[i] for i in range(candidate_size) if int(s[i]) == 1]
    # print('result', result)
    return result

def BOIM_vanilla_cluster_nngp(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    data = load_data("Cora", transform=transition_matrix())
    # data = load_data("CiteSeer", transform=transition_matrix())

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidates = [node for node, _ in deg[:candidate_size]]

    clusters = cluster_candidate_sets(candidate_sets)
    num_clusters = len(clusters)

    train_X = []
    train_Y = []
    source_sets = []

    initial_sample_size = 23
    clusters_to_sample = min(num_clusters, max(1, initial_sample_size // num_clusters))

    if clusters_to_sample > num_clusters:
        clusters_to_sample = num_clusters

    selected_clusters = np.random.choice(list(clusters.keys()), size=clusters_to_sample, replace=False)
    initial_samples = [index for cluster in selected_clusters for index in clusters[cluster]]
    initial_samples = np.random.choice(initial_samples, size=min(initial_sample_size, len(initial_samples)), replace=False)

    for index in initial_samples:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input, dtype=torch.double)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y, dtype=torch.double)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnngp_model = GNNGP(data, L=2, sigma_b=0.751087517315013, sigma_w=1.301961883639593, device=device)

    epsilon = torch.logspace(-3, 1, 101, device=device)
    gnngp_model.predict(epsilon)

    wrapped_gnngp_model = GNNGPBoTorchWrapper(gnngp_model, epsilon)

    acq_func = UpperConfidenceBound(model=wrapped_gnngp_model, beta=1.5489486601510272)
    function_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        clusters_to_sample = min(num_clusters, max(1, initial_sample_size // num_clusters))

        if clusters_to_sample > num_clusters:
            clusters_to_sample = num_clusters

        selected_clusters = np.random.choice(list(clusters.keys()), size=clusters_to_sample, replace=False)
        samples = [index for cluster in selected_clusters for index in clusters[cluster]]
        samples = np.random.choice(samples, size=min(initial_sample_size, len(samples)), replace=False)

        candidate_inputs = []
        for i in samples:
            candidate_set = candidate_sets[i]
            candidate_input = [1 if item in candidate_set else 0 for item in candidates]
            candidate_inputs.append(torch.tensor(candidate_input, dtype=torch.double))

        candidate_inputs = torch.stack(candidate_inputs)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=candidate_inputs,
        )

        found_candidate = candidate[0]

        signal = found_candidate.tolist()
        selected = [candidates[i] for i in range(candidate_size) if int(signal[i]) == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        max_train_Y_values.append(train_Y.max().item())

        try:
            gnngp_model.predict(epsilon)
        except RuntimeError as e:
            print(f"Error at iteration {iteration}:")
            print(f"train_X shape: {train_X.shape}, train_Y shape: {train_Y.shape}")
            print(f"train_X:\n{train_X}")
            print(f"train_Y:\n{train_Y}")
            print(f"Model state dict: {gnngp_model.state_dict()}")
            raise e

    best_index = train_Y.argmax().item()
    s = list(train_X[best_index])
    result = [candidates[i] for i in range(candidate_size) if int(s[i]) == 1]

    return result


def BOIM_vanilla_stratified_nngp(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    data = load_data("Cora", transform=transition_matrix())
    # data = load_data("CiteSeer", transform=transition_matrix())

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidates = [node for node, _ in deg[:candidate_size]]

    strata = stratify_candidate_sets(candidate_sets)
    num_strata = len(strata)

    train_X = []
    train_Y = []
    source_sets = []

    initial_sample_size = 23
    samples_per_stratum = max(1, initial_sample_size // num_strata)

    for stratum in strata.values():
        stratum_samples = np.random.choice(stratum, size=min(samples_per_stratum, len(stratum)), replace=False)
        for index in stratum_samples:
            source_set = candidate_sets[index]

            if diffusion_model == 'ic':
                e, _ = effectIC(G, config, source_set, num_of_sims)
            elif diffusion_model == 'lt':
                e, _ = effectLT(G, config, source_set, num_of_sims)
            else:
                raise NotImplementedError("Diffusion model not recognized.")
                
            input = [1 if item in source_set else 0 for item in candidates]
            input = torch.tensor(input, dtype=torch.double)

            train_X.append(input)
            train_Y.append([float(e)])
            source_sets.append(source_set)

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y, dtype=torch.double)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnngp_model = GNNGP(data, L=2, sigma_b=0.751087517315013, sigma_w=1.301961883639593, device=device)

    epsilon = torch.logspace(-3, 1, 101, device=device)
    gnngp_model.predict(epsilon)

    wrapped_gnngp_model = GNNGPBoTorchWrapper(gnngp_model, epsilon)

    acq_func = UpperConfidenceBound(model=wrapped_gnngp_model, beta=1.5489486601510272)
    function_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        samples_per_stratum = max(1, initial_sample_size // num_strata)
        samples = []

        for stratum in strata.values():
            stratum_samples = np.random.choice(stratum, size=min(samples_per_stratum, len(stratum)), replace=False)
            samples.extend([candidate_sets[i] for i in stratum_samples])

        candidate_inputs = []
        for source_set in samples:
            candidate_input = [1 if item in source_set else 0 for item in candidates]
            candidate_inputs.append(torch.tensor(candidate_input, dtype=torch.double))

        candidate_inputs = torch.stack(candidate_inputs)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=candidate_inputs,
        )

        found_candidate = candidate[0]

        signal = found_candidate.tolist()
        selected = [candidates[i] for i in range(candidate_size) if int(signal[i]) == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        max_train_Y_values.append(train_Y.max().item())

        try:
            gnngp_model.predict(epsilon)
        except RuntimeError as e:
            print(f"Error at iteration {iteration}:")
            print(f"train_X shape: {train_X.shape}, train_Y shape: {train_Y.shape}")
            print(f"train_X:\n{train_X}")
            print(f"train_Y:\n{train_Y}")
            print(f"Model state dict: {gnngp_model.state_dict()}")
            raise e

    best_index = train_Y.argmax().item()
    s = list(train_X[best_index])
    result = [candidates[i] for i in range(candidate_size) if int(s[i]) == 1]

    return result

def BOIM_vanilla_normal_nngp(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    data = load_data("Cora", transform=transition_matrix())
    # data = load_data("CiteSeer", transform=transition_matrix())

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    num_candidate_sets = len(candidate_sets)
    
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidates = [node for node, _ in deg[:candidate_size]]

    mean = (num_candidate_sets - 1) / 2
    std_dev = num_candidate_sets / 6  

    train_X = []
    train_Y = []
    source_sets = []

    initial_sample_size = 23
    initial_indices = set()
    while len(initial_indices) < initial_sample_size:
        new_index = int(np.clip(np.random.normal(mean, std_dev), 0, num_candidate_sets - 1))
        initial_indices.add(new_index)

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input, dtype=torch.double)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y, dtype=torch.double)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnngp_model = GNNGP(data, L=2, sigma_b=0.751087517315013, sigma_w=1.301961883639593, device=device)

    epsilon = torch.logspace(-3, 1, 101, device=device)
    gnngp_model.predict(epsilon)

    wrapped_gnngp_model = GNNGPBoTorchWrapper(gnngp_model, epsilon)

    acq_func = UpperConfidenceBound(model=wrapped_gnngp_model, beta=1.5489486601510272)
    function_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        sample_size = 23
        indices = set()
        while len(indices) < sample_size:
            new_index = int(np.clip(np.random.normal(mean, std_dev), 0, num_candidate_sets - 1))
            indices.add(new_index)

        samples = [candidate_sets[i] for i in indices]
        candidate_inputs = []
        for source_set in samples:
            candidate_input = [1 if item in source_set else 0 for item in candidates]
            candidate_inputs.append(torch.tensor(candidate_input, dtype=torch.double))

        candidate_inputs = torch.stack(candidate_inputs)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=candidate_inputs,
        )

        found_candidate = candidate[0]

        signal = found_candidate.tolist()
        selected = [candidates[i] for i in range(candidate_size) if int(signal[i]) == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        max_train_Y_values.append(train_Y.max().item())

        try:
            gnngp_model.predict(epsilon)
        except RuntimeError as e:
            print(f"Error at iteration {iteration}:")
            print(f"train_X shape: {train_X.shape}, train_Y shape: {train_Y.shape}")
            print(f"train_X:\n{train_X}")
            print(f"train_Y:\n{train_Y}")
            print(f"Model state dict: {gnngp_model.state_dict()}")
            raise e

    best_index = train_Y.argmax().item()
    s = list(train_X[best_index])
    result = [candidates[i] for i in range(candidate_size) if int(s[i]) == 1]

    return result

def BOIM_vanilla_normal_ntk(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    num_candidate_sets = len(candidate_sets)

    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidates = [node for node, _ in deg[:candidate_size]]

    mean = (num_candidate_sets - 1) / 2
    std_dev = num_candidate_sets / 6  

    train_X = []
    train_Y = []
    source_sets = []

    initial_sample_size = 23
    initial_indices = set()
    while len(initial_indices) < initial_sample_size:
        new_index = int(np.clip(np.random.normal(mean, std_dev), 0, num_candidate_sets - 1))
        initial_indices.add(new_index)

    for index in initial_indices:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input, dtype=torch.double)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y, dtype=torch.double)

    # Define the NTK model
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(128), stax.Relu(), stax.Dense(128), stax.Relu(), stax.Dense(train_X.size(1))  # Match last layer to input size
    )

    ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
    ntk_kernel = np.array(ntk_kernel)
    
    epsilon = torch.logspace(-3, 1, 101, device='cpu')
    wrapped_ntk_model = NTKBoTorchWrapper(ntk_kernel, epsilon)

    acq_func = ExpectedImprovement(model=wrapped_ntk_model, best_f=train_Y.max().item())

    function_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        sample_size = 23
        indices = set()
        while len(indices) < sample_size:
            new_index = int(np.clip(np.random.normal(mean, std_dev), 0, num_candidate_sets - 1))
            indices.add(new_index)

        samples = [candidate_sets[i] for i in indices]
        candidate_inputs = []
        for source_set in samples:
            candidate_input = [1 if item in source_set else 0 for item in candidates]
            candidate_inputs.append(torch.tensor(candidate_input, dtype=torch.double))

        candidate_inputs = torch.stack(candidate_inputs)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=candidate_inputs,
        )

        found_candidate = candidate[0]

        signal = found_candidate.tolist()
        selected = [candidates[i] for i in range(candidate_size) if int(signal[i]) == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        max_train_Y_values.append(train_Y.max().item())

        try:
            ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
            ntk_kernel = np.array(ntk_kernel)
            wrapped_ntk_model.ntk_kernel = torch.tensor(ntk_kernel, dtype=torch.float32)
        except RuntimeError as e:
            print(f"Error at iteration {iteration}:")
            print(f"train_X shape: {train_X.shape}, train_Y shape: {train_Y.shape}")
            print(f"train_X:\n{train_X}")
            print(f"train_Y:\n{train_Y}")
            raise e

    best_index = train_Y.argmax().item()
    s = list(train_X[best_index])
    result = [candidates[i] for i in range(candidate_size) if int(s[i]) == 1]

    return result

def BOIM_vanilla_cluster_ntk(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    num_candidate_sets = len(candidate_sets)

    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidates = [node for node, _ in deg[:candidate_size]]

    clusters = cluster_candidate_sets(candidate_sets)
    num_clusters = len(clusters)

    train_X = []
    train_Y = []
    source_sets = []

    initial_sample_size = 23
    clusters_to_sample = min(num_clusters, max(1, initial_sample_size // num_clusters))

    if clusters_to_sample > num_clusters:
        clusters_to_sample = num_clusters
    
    selected_clusters = np.random.choice(list(clusters.keys()), size=clusters_to_sample, replace=False)
    initial_samples = [index for cluster in selected_clusters for index in clusters[cluster]]
    initial_samples = np.random.choice(initial_samples, size=min(initial_sample_size, len(initial_samples)), replace=False)

    for index in initial_samples:
        source_set = candidate_sets[index]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, source_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, source_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")
            
        input = [1 if item in source_set else 0 for item in candidates]
        input = torch.tensor(input, dtype=torch.double)

        train_X.append(input)
        train_Y.append([float(e)])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    # Define the NTK model
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(128), stax.Relu(), stax.Dense(128), stax.Relu(), stax.Dense(train_X.size(1))  # Match last layer to input size
    )

    ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
    ntk_kernel = np.array(ntk_kernel)
    
    epsilon = torch.logspace(-3, 1, 101, device='cpu')
    wrapped_ntk_model = NTKBoTorchWrapper(ntk_kernel, epsilon)

    acq_func = ExpectedImprovement(model=wrapped_ntk_model, best_f=train_Y.max().item())

    function_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        clusters_to_sample = min(num_clusters, max(1, initial_sample_size // num_clusters))

        if clusters_to_sample > num_clusters:
            clusters_to_sample = num_clusters

        selected_clusters = np.random.choice(list(clusters.keys()), size=clusters_to_sample, replace=False)
        samples = [index for cluster in selected_clusters for index in clusters[cluster]]
        samples = np.random.choice(samples, size=min(initial_sample_size, len(samples)), replace=False)

        candidate_inputs = []
        for i in samples:
            candidate_set = candidate_sets[i]
            candidate_input = [1 if item in candidate_set else 0 for item in candidates]
            candidate_inputs.append(torch.tensor(candidate_input, dtype=torch.double))

        candidate_inputs = torch.stack(candidate_inputs)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=candidate_inputs,
        )

        found_candidate = candidate[0]

        signal = found_candidate.tolist()
        selected = [candidates[i] for i in range(candidate_size) if int(signal[i]) == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        max_train_Y_values.append(train_Y.max().item())

        try:
            ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
            ntk_kernel = np.array(ntk_kernel)
            wrapped_ntk_model.ntk_kernel = torch.tensor(ntk_kernel, dtype=torch.float32)
        except RuntimeError as e:
            print(f"Error at iteration {iteration}:")
            print(f"train_X shape: {train_X.shape}, train_Y shape: {train_Y.shape}")
            print(f"train_X:\n{train_X}")
            print(f"train_Y:\n{train_Y}")
            raise e

    best_index = train_Y.argmax().item()
    s = list(train_X[best_index])
    result = [candidates[i] for i in range(candidate_size) if int(s[i]) == 1]

    return result

def BOIM_vanilla_stratified_ntk(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources):
    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources)
    num_candidate_sets = len(candidate_sets)

    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidates = [node for node, _ in deg[:candidate_size]]

    strata = stratify_candidate_sets(candidate_sets)
    num_strata = len(strata)

    train_X = []
    train_Y = []
    source_sets = []

    initial_sample_size = 23
    samples_per_stratum = max(1, initial_sample_size // num_strata)
    
    for stratum in strata.values():
        stratum_samples = np.random.choice(stratum, size=min(samples_per_stratum, len(stratum)), replace=False)
        for index in stratum_samples:
            source_set = candidate_sets[index]

            if diffusion_model == 'ic':
                e, _ = effectIC(G, config, source_set, num_of_sims)
            elif diffusion_model == 'lt':
                e, _ = effectLT(G, config, source_set, num_of_sims)
            else:
                raise NotImplementedError("Diffusion model not recognized.")
                
            input = [1 if item in source_set else 0 for item in candidates]
            input = torch.tensor(input, dtype=torch.double)

            train_X.append(input)
            train_Y.append([float(e)])
            source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    # Define the NTK model
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(128), stax.Relu(), stax.Dense(128), stax.Relu(), stax.Dense(train_X.size(1))  # Match last layer to input size
    )

    ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
    ntk_kernel = np.array(ntk_kernel)
    
    epsilon = torch.logspace(-3, 1, 101, device='cpu')
    wrapped_ntk_model = NTKBoTorchWrapper(ntk_kernel, epsilon)

    acq_func = ExpectedImprovement(model=wrapped_ntk_model, best_f=train_Y.max().item())

    function_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        samples_per_stratum = max(1, initial_sample_size // num_strata)
        samples = []
        
        for stratum in strata.values():
            stratum_samples = np.random.choice(stratum, size=min(samples_per_stratum, len(stratum)), replace=False)
            samples.extend([candidate_sets[i] for i in stratum_samples])

        candidate_inputs = []
        for source_set in samples:
            candidate_input = [1 if item in source_set else 0 for item in candidates]
            candidate_inputs.append(torch.tensor(candidate_input, dtype=torch.double))

        candidate_inputs = torch.stack(candidate_inputs)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=candidate_inputs,
        )

        found_candidate = candidate[0]

        signal = found_candidate.tolist()
        selected = [candidates[i] for i in range(candidate_size) if int(signal[i]) == 1]

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        max_train_Y_values.append(train_Y.max().item())

        try:
            ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
            ntk_kernel = np.array(ntk_kernel)
            wrapped_ntk_model.ntk_kernel = torch.tensor(ntk_kernel, dtype=torch.float32)
        except RuntimeError as e:
            print(f"Error at iteration {iteration}:")
            print(f"train_X shape: {train_X.shape}, train_Y shape: {train_Y.shape}")
            print(f"train_X:\n{train_X}")
            print(f"train_Y:\n{train_Y}")
            raise e

    best_index = train_Y.argmax().item()
    s = list(train_X[best_index])
    result = [candidates[i] for i in range(candidate_size) if int(s[i]) == 1]

    return result

def BOIM_fourier_no_GSS_random_ntk(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance):
    nl = nx.normalized_laplacian_matrix(G)
    _, eig_vect = np.linalg.eigh(nl.todense())
    UT = np.linalg.inv(eig_vect)
    UT_inv = eig_vect

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    
    # Define candidate nodes based on degree
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidate_nodes = [node for node, _ in deg[:candidate_size]]

    num_candidate_sets = len(candidate_sets)
    mean = (num_candidate_sets - 1) / 2
    std_dev = num_candidate_sets / 6

    train_X = []
    train_Y = []

    num_initial_samples = 23  # Ensure num_initial_samples is defined
    initial_indices = set()
    while len(initial_indices) < num_initial_samples:
        new_index = int(np.clip(np.random.normal(mean, std_dev), 0, num_candidate_sets - 1))
        initial_indices.add(new_index)

    for index in initial_indices:
        selected_set = candidate_sets[index]
        selected_signal = create_signal_from_source_set(G, selected_set, UT)
        # print(f"Selected set: {selected_set}")
        # print(f"Selected signal: {selected_signal}")

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y, dtype=torch.double)
    # print(f"Initial train_X: {train_X}")
    # print(f"Initial train_Y: {train_Y}")

    # Define the NTK model
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(128), stax.Relu(), stax.Dense(128), stax.Relu(), stax.Dense(train_X.size(1))  # Match last layer to input size
    )

    ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
    ntk_kernel = np.array(ntk_kernel)
    
    epsilon = torch.logspace(-3, 1, 101, device='cpu')
    wrapped_ntk_model = NTKBoTorchWrapper(ntk_kernel, epsilon)

    acq_func = ExpectedImprovement(model=wrapped_ntk_model, best_f=train_Y.max().item())

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        sample_size = 23
        indices = set()
        while len(indices) < sample_size:
            new_index = int(np.clip(np.random.normal(mean, std_dev), 0, num_candidate_sets - 1))
            indices.add(new_index)

        samples = [candidate_sets[i] for i in indices]
        candidate_inputs = []
        for source_set in samples:
            candidate_input = create_signal_from_source_set(G, source_set, UT)
            candidate_inputs.append(torch.tensor(candidate_input, dtype=torch.double))

        candidate_inputs = torch.stack(candidate_inputs)
        # print(f"Iteration {iteration}: candidate_inputs: {candidate_inputs}")

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=candidate_inputs,
        )

        found_candidate = candidate[0]

        signal = found_candidate.tolist()
        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)
        # print(f"Iteration {iteration}: selected: {selected}")

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        acquisition_values.append(acq_value[0].item())  # Append the first element as a scalar
        max_train_Y_values.append(train_Y.max().item())
        # print(f"Iteration {iteration}: new_Y: {new_Y}")

        try:
            ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
            ntk_kernel = np.array(ntk_kernel)
            wrapped_ntk_model.ntk_kernel = torch.tensor(ntk_kernel, dtype=torch.float32)
        except RuntimeError as e:
            print(f"Error at iteration {iteration}:")
            print(f"train_X shape: {train_X.shape}, train_Y shape: {train_Y.shape}")
            print(f"train_X:\n{train_X}")
            print(f"train_Y:\n{train_Y}")
            raise e

    best_index = train_Y.argmax().item()
    s = train_X[best_index].numpy()
    # print(f"Signal of best index: {s}")
    result = find_source_set_from_fourier(s, number_of_sources, UT_inv)
    print(f"Best result: {result}")
    return result


def BOIM_fourier_no_GSS_normal_ntk(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance):
    nl = nx.normalized_laplacian_matrix(G)
    _, eig_vect = np.linalg.eigh(nl.todense())
    UT = np.linalg.inv(eig_vect)
    UT_inv = eig_vect

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    
    # Define candidate nodes based on degree
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidate_nodes = [node for node, _ in deg[:candidate_size]]

    num_candidate_sets = len(candidate_sets)
    mean = (num_candidate_sets - 1) / 2
    std_dev = num_candidate_sets / 6

    train_X = []
    train_Y = []

    num_initial_samples = 23  # Ensure num_initial_samples is defined
    initial_indices = set()
    while len(initial_indices) < num_initial_samples:
        new_index = int(np.clip(np.random.normal(mean, std_dev), 0, num_candidate_sets - 1))
        initial_indices.add(new_index)

    for index in initial_indices:
        selected_set = candidate_sets[index]
        selected_signal = create_signal_from_source_set(G, selected_set, UT)
        # print(f"Selected set: {selected_set}")
        # print(f"Selected signal: {selected_signal}")

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y, dtype=torch.double)
    # print(f"Initial train_X: {train_X}")
    # print(f"Initial train_Y: {train_Y}")

    # Define the NTK model
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(128), stax.Relu(), stax.Dense(128), stax.Relu(), stax.Dense(train_X.size(1))  # Match last layer to input size
    )

    ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
    ntk_kernel = np.array(ntk_kernel)
    
    epsilon = torch.logspace(-3, 1, 101, device='cpu')
    wrapped_ntk_model = NTKBoTorchWrapper(ntk_kernel, epsilon)

    acq_func = ExpectedImprovement(model=wrapped_ntk_model, best_f=train_Y.max().item())

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        sample_size = 23
        indices = set()
        while len(indices) < sample_size:
            new_index = int(np.clip(np.random.normal(mean, std_dev), 0, num_candidate_sets - 1))
            indices.add(new_index)

        samples = [candidate_sets[i] for i in indices]
        candidate_inputs = []
        for source_set in samples:
            candidate_input = create_signal_from_source_set(G, source_set, UT)
            candidate_inputs.append(torch.tensor(candidate_input, dtype=torch.double))

        candidate_inputs = torch.stack(candidate_inputs)
        # print(f"Iteration {iteration}: candidate_inputs: {candidate_inputs}")

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=candidate_inputs,
        )

        found_candidate = candidate[0]

        signal = found_candidate.tolist()
        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)
        # print(f"Iteration {iteration}: selected: {selected}")

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        acquisition_values.append(acq_value[0].item())  # Append the first element as a scalar
        max_train_Y_values.append(train_Y.max().item())
        # print(f"Iteration {iteration}: new_Y: {new_Y}")

        try:
            ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
            ntk_kernel = np.array(ntk_kernel)
            wrapped_ntk_model.ntk_kernel = torch.tensor(ntk_kernel, dtype=torch.float32)
        except RuntimeError as e:
            print(f"Error at iteration {iteration}:")
            print(f"train_X shape: {train_X.shape}, train_Y shape: {train_Y.shape}")
            print(f"train_X:\n{train_X}")
            print(f"train_Y:\n{train_Y}")
            raise e

    best_index = train_Y.argmax().item()
    s = train_X[best_index].numpy()
    # print(f"Signal of best index: {s}")
    result = find_source_set_from_fourier(s, number_of_sources, UT_inv)
    print(f"Best result: {result}")
    return result

def BOIM_fourier_no_GSS_cluster_ntk(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance):

    nl = nx.normalized_laplacian_matrix(G)
    _, eig_vect = np.linalg.eigh(nl.todense())
    UT = np.linalg.inv(eig_vect)
    UT_inv = eig_vect

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    
    # Define candidate nodes based on degree
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidate_nodes = [node for node, _ in deg[:candidate_size]]

    clusters = cluster_candidate_sets(candidate_sets)
    num_clusters = len(clusters)

    train_X = []
    train_Y = []

    num_initial_samples = 23
    clusters_to_sample = min(num_clusters, max(1, num_initial_samples // num_clusters))

    if clusters_to_sample > num_clusters:
        clusters_to_sample = num_clusters

    selected_clusters = np.random.choice(list(clusters.keys()), size=clusters_to_sample, replace=False)
    initial_samples = [index for cluster in selected_clusters for index in clusters[cluster]]
    initial_samples = np.random.choice(initial_samples, size=min(num_initial_samples, len(initial_samples)), replace=False)

    for index in initial_samples:
        selected_set = candidate_sets[index]
        selected_signal = create_signal_from_source_set(G, selected_set, UT)
        # print(f"Selected set: {selected_set}")
        # print(f"Selected signal: {selected_signal}")

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected_set, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected_set, num_of_sims)
        else:
            raise NotImplementedError("Diffusion model not recognized.")

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y, dtype=torch.double)
    # print(f"Initial train_X: {train_X}")
    # print(f"Initial train_Y: {train_Y}")

    # Define the NTK model
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(128), stax.Relu(), stax.Dense(128), stax.Relu(), stax.Dense(train_X.size(1))  # Match last layer to input size
    )

    ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
    ntk_kernel = np.array(ntk_kernel)
    
    epsilon = torch.logspace(-3, 1, 101, device='cpu')
    wrapped_ntk_model = NTKBoTorchWrapper(ntk_kernel, epsilon)

    acq_func = ExpectedImprovement(model=wrapped_ntk_model, best_f=train_Y.max().item())

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        clusters_to_sample = min(num_clusters, max(1, num_initial_samples // num_clusters))

        if clusters_to_sample > num_clusters:
            clusters_to_sample = num_clusters

        selected_clusters = np.random.choice(list(clusters.keys()), size=clusters_to_sample, replace=False)
        samples = [index for cluster in selected_clusters for index in clusters[cluster]]
        samples = np.random.choice(samples, size=min(num_initial_samples, len(samples)), replace=False)

        candidate_inputs = []
        for i in samples:
            candidate_set = candidate_sets[i]
            candidate_input = create_signal_from_source_set(G, candidate_set, UT)
            candidate_inputs.append(torch.tensor(candidate_input, dtype=torch.double))

        candidate_inputs = torch.stack(candidate_inputs)
        # print(f"Iteration {iteration}: candidate_inputs: {candidate_inputs}")

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=candidate_inputs,
        )

        found_candidate = candidate[0]

        signal = found_candidate.tolist()
        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)
        # print(f"Iteration {iteration}: selected: {selected}")

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        acquisition_values.append(acq_value[0].item())  # Append the first element as a scalar
        max_train_Y_values.append(train_Y.max().item())
        # print(f"Iteration {iteration}: new_Y: {new_Y}")

        try:
            ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
            ntk_kernel = np.array(ntk_kernel)
            wrapped_ntk_model.ntk_kernel = torch.tensor(ntk_kernel, dtype=torch.float32)
        except RuntimeError as e:
            print(f"Error at iteration {iteration}:")
            print(f"train_X shape: {train_X.shape}, train_Y shape: {train_Y.shape}")
            print(f"train_X:\n{train_X}")
            print(f"train_Y:\n{train_Y}")
            raise e

    best_index = train_Y.argmax().item()
    s = train_X[best_index].numpy()
    # print(f"Signal of best index: {s}")
    result = find_source_set_from_fourier(s, number_of_sources, UT_inv)
    print(f"Best result: {result}")
    return result


def BOIM_fourier_no_GSS_stratified_ntk(G, config, num_iterations, num_of_sims, candidate_size, diffusion_model, number_of_sources, allowed_shortest_distance):

    nl = nx.normalized_laplacian_matrix(G)
    _, eig_vect = np.linalg.eigh(nl.todense())
    UT = np.linalg.inv(eig_vect)
    UT_inv = eig_vect

    candidate_sets = create_candidate_set_pool_filtering(G, candidate_size, number_of_sources, allowed_shortest_distance)
    
    # Define candidate nodes based on degree
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)
    candidate_nodes = [node for node, _ in deg[:candidate_size]]

    strata = stratify_candidate_sets(candidate_sets)
    num_strata = len(strata)

    train_X = []
    train_Y = []

    num_initial_samples = 23
    samples_per_stratum = max(1, num_initial_samples // num_strata)
    
    for stratum in strata.values():
        stratum_samples = np.random.choice(stratum, size=min(samples_per_stratum, len(stratum)), replace=False)
        for index in stratum_samples:
            selected_set = candidate_sets[index]
            selected_signal = create_signal_from_source_set(G, selected_set, UT)
            # print(f"Selected set: {selected_set}")
            # print(f"Selected signal: {selected_signal}")

            if diffusion_model == 'ic':
                e, _ = effectIC(G, config, selected_set, num_of_sims)
            elif diffusion_model == 'lt':
                e, _ = effectLT(G, config, selected_set, num_of_sims)
            else:
                raise NotImplementedError("Diffusion model not recognized.")

            input = torch.FloatTensor(selected_signal)

            train_X.append(input)
            train_Y.append([float(e)])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y, dtype=torch.double)
    # print(f"Initial train_X: {train_X}")
    # print(f"Initial train_Y: {train_Y}")

    # Define the NTK model
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(128), stax.Relu(), stax.Dense(128), stax.Relu(), stax.Dense(train_X.size(1))  # Match last layer to input size
    )

    ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
    ntk_kernel = np.array(ntk_kernel)
    
    epsilon = torch.logspace(-3, 1, 101, device='cpu')
    wrapped_ntk_model = NTKBoTorchWrapper(ntk_kernel, epsilon)

    acq_func = ExpectedImprovement(model=wrapped_ntk_model, best_f=train_Y.max().item())

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        samples_per_stratum = max(1, num_initial_samples // num_strata)
        samples = []
        
        for stratum in strata.values():
            stratum_samples = np.random.choice(stratum, size=min(samples_per_stratum, len(stratum)), replace=False)
            samples.extend([candidate_sets[i] for i in stratum_samples])

        candidate_inputs = []
        for source_set in samples:
            candidate_input = create_signal_from_source_set(G, source_set, UT)
            candidate_inputs.append(torch.tensor(candidate_input, dtype=torch.double))

        candidate_inputs = torch.stack(candidate_inputs)
        # print(f"Iteration {iteration}: candidate_inputs: {candidate_inputs}")

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,
            choices=candidate_inputs,
        )

        found_candidate = candidate[0]

        signal = found_candidate.tolist()
        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)
        # print(f"Iteration {iteration}: selected: {selected}")

        if diffusion_model == 'ic':
            e, _ = effectIC(G, config, selected, num_of_sims)
        elif diffusion_model == 'lt':
            e, _ = effectLT(G, config, selected, num_of_sims)

        new_Y = torch.tensor([[float(e)]], dtype=torch.double)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y], dim=0)

        function_values.append(new_Y.item())
        acquisition_values.append(acq_value[0].item())  # Append the first element as a scalar
        max_train_Y_values.append(train_Y.max().item())
        # print(f"Iteration {iteration}: new_Y: {new_Y}")

        try:
            ntk_kernel = kernel_fn(train_X.numpy(), train_X.numpy(), 'ntk').block_until_ready()
            ntk_kernel = np.array(ntk_kernel)
            wrapped_ntk_model.ntk_kernel = torch.tensor(ntk_kernel, dtype=torch.float32)
        except RuntimeError as e:
            print(f"Error at iteration {iteration}:")
            print(f"train_X shape: {train_X.shape}, train_Y shape: {train_Y.shape}")
            print(f"train_X:\n{train_X}")
            print(f"train_Y:\n{train_Y}")
            raise e

    best_index = train_Y.argmax().item()
    s = train_X[best_index].numpy()
    # print(f"Signal of best index: {s}")
    result = find_source_set_from_fourier(s, number_of_sources, UT_inv)
    print(f"Best result: {result}")
    return result
