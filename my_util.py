import sys
import math
import time
import ndlib
import random
import logging
import gpytorch
import warnings
import numpy as np
import pandas as pd
import heapdict as hd
import networkx as nx
import torch.nn as nn
from pyDOE import lhs
import statistics as s
import itertools as it
import scipy.sparse as sp
from graphGeneration import *
from scipy.linalg import expm
from scipy.sparse import diags
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import Counter
from scipy.sparse import issparse
from gpytorch.module import Module
import ndlib.models.epidemics as ep
from scipy.sparse import csc_matrix
from gpytorch.means.mean import Mean
from joblib import Parallel, delayed
from scipy.sparse.linalg import eigsh
import community as community_louvain
from torch_geometric.data import Data
import ndlib.models.ModelConfig as mc
from scipy.interpolate import interp2d
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from torch_geometric.nn.inits import reset
from botorch.fit import fit_gpytorch_model
from hodgelaplacians import HodgeLaplacians
from gpytorch.models.exact_gp import ExactGP
from botorch.optim import optimize_acqf_discrete
from concurrent.futures import ThreadPoolExecutor
from gpytorch.kernels import RBFKernel, RFFKernel
from botorch.acquisition import ExpectedImprovement
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from sklearn.cluster import KMeans, SpectralClustering
from typing import Any, List, NoReturn, Optional, Union
from gpytorch.constraints.constraints import GreaterThan
from torch_geometric.nn import GCNConv, global_mean_pool
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import Log, OutcomeTransform
from botorch.models import SaasFullyBayesianSingleTaskGP
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.utils import fantasize as fantasize_flag, validate_input_scaling
from random import uniform, seed
import operator
import copy
from torch_geometric.utils import to_networkx
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from botorch.acquisition import UpperConfidenceBound
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive
from gpytorch.means import ZeroMean
from gnngp import GNNGP
from datasets import load_data, transition_matrix
from jax import numpy as jnp
from jax import random as jax_random
from neural_tangents import stax

################################################
# Global parameters
################################################
diffusion_model = "ic" # "ic" or "lt"
graph_size = 1000
candidate_size = 50 # candidate pool size
number_of_sources = 3 # budget for IM
num_iterations = 300 # budget for BO
actual_time_step_size = 5 # diffusion parameter
allowed_shortest_distance = 1 # shortest distance between sources for filtering
num_of_sims = 10
number_of_clusters = 20
num_initial_samples = 20


def get_gaussian_likelihood_with_gamma_prior(
    batch_shape: Optional[torch.Size] = None,
) -> GaussianLikelihood:
    r"""Constructs the GaussianLikelihood that is used by default by
    several models. This uses a Gamma(1.1, 0.05) prior and constrains the
    noise level to be greater than MIN_INFERRED_NOISE_LEVEL (=1e-4).
    """
    batch_shape = torch.Size() if batch_shape is None else batch_shape
    noise_prior = GammaPrior(1.1, 0.05)
    noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
    return GaussianLikelihood(
        noise_prior=noise_prior,
        batch_shape=batch_shape,
        noise_constraint=GreaterThan(
            1e-4,
            transform=None,
            initial_value=noise_prior_mode,
        ),
    )

# define a class inherited from SingleTaskGP for RBF/rff kernel
class RBFSingleTaskGP(SingleTaskGP):

    def __init__(self, train_X: torch.Tensor,
        train_Y: torch.Tensor,
        likelihood: Optional[Likelihood] = None,
        covar_module: Optional[Module] = None,
        mean_module: Optional[Mean] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        r"""
        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x m` tensor of training observations.
            likelihood: A likelihood. If omitted, use a standard
                GaussianLikelihood with inferred noise level.
            covar_module: The module computing the covariance (Kernel) matrix.
                If omitted, use a `RBF`.
            mean_module: The mean function to be used. If omitted, use a
                `ConstantMean`.
            outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
            input_transform: An input transform that is applied in the model's
                forward pass.
        """
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)
        self._validate_tensor_args(X=transformed_X, Y=train_Y)
        ignore_X_dims = getattr(self, "_ignore_X_dims_scaling_check", None)
        validate_input_scaling(
            train_X=transformed_X, train_Y=train_Y, ignore_X_dims=ignore_X_dims
        )
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        train_X, train_Y, _ = self._transform_tensor_args(X=train_X, Y=train_Y)
        if likelihood is None:
            likelihood = get_gaussian_likelihood_with_gamma_prior(
                batch_shape=self._aug_batch_shape
            )
        else:
            self._is_custom_likelihood = True
        ExactGP.__init__(
            self, train_inputs=train_X, train_targets=train_Y, likelihood=likelihood
        )
        if mean_module is None:
            mean_module = ConstantMean(batch_shape=self._aug_batch_shape)
        self.mean_module = mean_module
        if covar_module is None:
            covar_module = RBFKernel()
            self._subset_batch_dict = {
                "likelihood.noise_covar.raw_noise": -2,
                "mean_module.raw_constant": -1,
                "covar_module.raw_outputscale": -1,
                "covar_module.base_kernel.raw_lengthscale": -3,
            }
        self.covar_module = covar_module
        # TODO: Allow subsetting of other covar modules
        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform
        self.to(train_X)

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# top candidate_size nodes with highest degree centrality as candidate pool

def create_candidate_set_pool_filtering(G, candidate_size=100, number_of_sources=3, allowed_shortest_distance=2):
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)

    candidates = []

    for candidate in deg[:candidate_size]:
        candidates.append(candidate[0])

    candidate_source_sets = []

    for selected_set in comb(candidates, number_of_sources):

        shortest_distance = 5
        for i in range(number_of_sources-2):
            start = selected_set[i]
            for j in range(i+1, number_of_sources-1):
                end = selected_set[j]
                distance = nx.shortest_path_length(G, source=start, target=end)
                if distance < shortest_distance:
                    shortest_distance = distance
        if shortest_distance > allowed_shortest_distance:            
            candidate_source_sets.append(selected_set)

    # check_candidate_set(candidate_source_sets)        
    
    return candidate_source_sets

def check_candidate_set(candidate_source_sets):
    candidate_sets = candidate_source_sets
    print("candidate sets", candidate_sets)
    logging.basicConfig(filename='candidate_sets_analysis.log', level=logging.INFO)
    logging.info("Analyzing candidate sets...")

    for i, set in enumerate(candidate_sets):
        logging.info(f"Set {i+1} size: {len(set)}")


def create_candidate_set_pool(G, candidate_size=100, number_of_sources=3):
    deg = sorted(G.degree, key=lambda x: x[1], reverse=True)

    candidates = []

    for candidate in deg[:candidate_size]:
        candidates.append(candidate[0])

    candidate_source_sets = []

    for selected_set in comb(candidates, number_of_sources):

        candidate_source_sets.append(selected_set)

    return candidate_source_sets


def fourier_transfer_for_all_candidate_set(candidate_sets, UT):

    n = len(UT)
    # print("n", n)

    signals = []
    for source_set in candidate_sets:
        a = [0 for i in range(n)]
        # print('a', a)
        for node in source_set:
            # print("node", node)
            a[node] = 1
        signal = np.matmul(a, UT)
        signals.append(signal)
    # print('signals',signals)
    return signals

def create_signal_from_source_set(G, sampled_set, UT):

    n = len(UT)

    a = [0 for i in range(n)]
    for node in sampled_set:
        a[node] = 1
    signal = np.matmul(a, UT)

    return signal

def find_source_set_from_fourier(signal, number_of_sources, UT_inv):

    source_set = []

    a = np.matmul(signal, UT_inv)
    b = np.around(a)
    for i in range(len(b)):
        if b[i] == 1:
            source_set.append(i)

    if len(source_set) != number_of_sources:
        raise NameError('length of source set is not the estimated number')

    return source_set

################################################
# Mostly for Sobol
################################################

def combinations(alist):
  n = len(alist)
  subs = [[]]

  for item in alist:
    subs += [curr + [item] for curr in subs]
  subs.sort(key=len)
  return subs

def subcombs(alist):
  subs = combinations(alist)
  subs.remove([])
  subs.remove(alist)
  subs.sort(key=len)
  return subs

def substract(alist, blist):
  a = []
  for i in alist:
    a.append(i)
  for i in blist:
    a.remove(i)
  return a

def diff(rank, order):
    n = len(rank)
    if (len(order) != n):
      print('the lengths do not match')
      pass
    else:
      difference = 0
      for i in range(n):
        item = rank[i]
        index = order.index(item)
        delta = abs(index - i)
        difference += delta
    return difference

def simulationIC(r, g, result, config):

    title = []
    for i in result:
        title.append(i)
    title.append('result')

    df = pd.DataFrame(columns=title)

    n = len(result)

    for combs in combinations(result):
        input = []
        for i in range(n):
            item = 1 if result[i] in combs else 0
            input.append(item)

        for i in range(r):

            input1 = []
            for item in input:
                input1.append(item)

            g_mid = g.__class__()
            g_mid.add_nodes_from(g)
            g_mid.add_edges_from(g.edges)

            model_mid = ep.IndependentCascadesModel(g_mid)
            config_mid = mc.Configuration()
            config_mid.add_model_initial_configuration('Infected', combs)

            for a, b in g_mid.edges():
                weight = config.config["edges"]['threshold'][(a, b)]
                g_mid[a][b]['weight'] = weight
                config_mid.add_edge_configuration('threshold', (a, b), weight)

            model_mid.set_initial_status(config_mid)

            iterations = model_mid.iteration_bunch(actual_time_step_size)
            trends = model_mid.build_trends(iterations)

            total_no = 0

            for j in range(actual_time_step_size):
                a = iterations[j]['node_count'][1]
                total_no += a

            input1.append(total_no)

            newdf = pd.DataFrame([input1], columns=title)

            df = pd.concat([df,newdf])
    return df

def simulationLT(r, g, result, config):

    title = []
    for i in result:
        title.append(i)
    title.append('result')

    df = pd.DataFrame(columns=title)

    n = len(result)

    for combs in combinations(result):
        input = []
        for i in range(n):
            item = 1 if result[i] in combs else 0
            input.append(item)

        for i in range(r):

            input1 = []
            for item in input:
                input1.append(item)

            g_mid = g.__class__()
            g_mid.add_nodes_from(g)
            g_mid.add_edges_from(g.edges)

            model_mid = ep.ThresholdModel(g_mid)
            config_mid = mc.Configuration()
            config_mid.add_model_initial_configuration('Infected', combs)

            for a, b in g_mid.edges():
                weight = config.config["edges"]['threshold'][(a, b)]
                g_mid[a][b]['weight'] = weight
                config_mid.add_edge_configuration('threshold', (a, b), weight)

            for i in g_mid.nodes():
                threshold = random.randrange(1, 20)
                threshold = round(threshold / 100, 2)
                config_mid.add_node_configuration("threshold", i, threshold)

            model_mid.set_initial_status(config_mid)

            iterations = model_mid.iteration_bunch(actual_time_step_size)
            trends = model_mid.build_trends(iterations)

            total_no = iterations[actual_time_step_size-1]['node_count'][1]
            input1.append(total_no)

            newdf = pd.DataFrame([input1], columns=title)

            df = pd.concat([df,newdf])
    return df

def SobolT(df, result):
    sobolt = {}

    for node in result:

        backup = []
        for item in result:
            backup.append(item)

        backup.remove(node)

        var = []

        for sub in combinations(backup):

            means = []

            for case in combinations([node]):

                total = []

                seeds = sub + case

                subdf = df

                for item in result:
                    if item in seeds:
                        a = (subdf[item] == 1)
                    else:
                        a = (subdf[item] == 0)

                    subdf = subdf[a]

                means.append(s.mean(subdf['result']))
            var.append(s.pvariance(means))

        sobolt[node] = s.mean(var)

    return sobolt

def effectIC(g, config, sources,rounds=10):

    input = []

    for i in range(rounds):

      g_mid = g.__class__()
      g_mid.add_nodes_from(g)
      g_mid.add_edges_from(g.edges)

      model_mid = ep.IndependentCascadesModel(g_mid)
      config_mid = mc.Configuration()
      config_mid.add_model_initial_configuration('Infected', sources)

      for a, b in g_mid.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_mid[a][b]['weight'] = weight
        config_mid.add_edge_configuration('threshold', (a, b), weight)

      model_mid.set_initial_status(config_mid)

      iterations = model_mid.iteration_bunch(actual_time_step_size)
      trends = model_mid.build_trends(iterations)

      total_no = 0

      for j in range(actual_time_step_size):
        a = iterations[j]['node_count'][1]
        total_no += a

      input.append(total_no)

    e = s.mean(input)
    v = s.stdev(input)

    return e,v

def effectLT(g, config, sources,rounds=10):

    input = []

    for i in range(rounds):

      g_mid = g.__class__()
      g_mid.add_nodes_from(g)
      g_mid.add_edges_from(g.edges)

      model_mid = ep.ThresholdModel(g_mid)
      config_mid = mc.Configuration()
      config_mid.add_model_initial_configuration('Infected', sources)

      for a, b in g_mid.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_mid[a][b]['weight'] = weight
        config_mid.add_edge_configuration('threshold', (a, b), weight)

      for i in g.nodes():
          threshold = random.randrange(1, 20)
          threshold = round(threshold / 100, 2)
          config_mid.add_node_configuration("threshold", i, threshold)

      model_mid.set_initial_status(config_mid)

      iterations = model_mid.iteration_bunch(actual_time_step_size)
      trends = model_mid.build_trends(iterations)

      total_no = iterations[actual_time_step_size-1]['node_count'][1]
      input.append(total_no)

    e = s.mean(input)
    v = s.stdev((input))

    return e,v

#######################################################
def compute_padding(G, alpha=0.5):
    simplices = generate_simplices(G, 1)
    hodge = HodgeLaplacians(simplices, maxdimension=1)
    L_0 = hodge.getHodgeLaplacian(d=0).toarray()
    L_1 = hodge.getHodgeLaplacian(d=1).toarray()

    max_dim = max(L_0.shape[0], L_1.shape[0])
    padded_L_0 = np.zeros((max_dim, max_dim))
    padded_L_1 = np.zeros((max_dim, max_dim))

    padded_L_0[:L_0.shape[0], :L_0.shape[1]] = L_0
    padded_L_1[:L_1.shape[0], :L_1.shape[1]] = L_1

    # Weighted combination of padded Hodge Laplacians
    L_combined = alpha * padded_L_0 + (1 - alpha) * padded_L_1

    return L_combined   

def compute_heat_kernel(L, t):
    """Compute the heat kernel for a given Laplacian matrix and time t."""
    return expm(-t * L)

def resize_matrix(smaller, target_dim):
    x = np.linspace(0, 1, num=smaller.shape[0])
    y = np.linspace(0, 1, num=smaller.shape[1])
    interp = interp2d(x, y, smaller, kind='cubic')
    
    new_x = np.linspace(0, 1, num=target_dim[0])
    new_y = np.linspace(0, 1, num=target_dim[1])
    return interp(new_x, new_y)

def combined_heat_kernel(G, t=1.0, alpha=0.5):
    simplices = generate_simplices(G, 1)
    hodge = HodgeLaplacians(simplices, maxdimension=1)
    L_0 = hodge.getHodgeLaplacian(d=0).toarray()
    L_1 = hodge.getHodgeLaplacian(d=1).toarray()

    K_0 = compute_heat_kernel(L_0, t)
    K_1 = compute_heat_kernel(L_1, t)

    if K_0.shape[0] > K_1.shape[0]:
        resized_K_1 = resize_matrix(K_1, K_0.shape)
        combined_kernel = alpha * K_0 + (1 - alpha) * resized_K_1
    else:
        resized_K_0 = resize_matrix(K_0, K_1.shape)
        combined_kernel = alpha * resized_K_0 + (1 - alpha) * K_1

    return combined_kernel

def combined_heat_kernel_nl_sparse(G, simplices, t=1.0, alpha=0.5):
    hodge = HodgeLaplacians(simplices, maxdimension=2)
    L_0 = hodge.getHodgeLaplacian(d=0)
    L_1 = hodge.getHodgeLaplacian(d=1)

    L_0 = normalize_laplacian_sparse(L_0).toarray()
    L_1 = normalize_laplacian_sparse(L_1).toarray()

    K_0 = compute_heat_kernel(L_0, t)
    K_1 = compute_heat_kernel(L_1, t)

    if K_0.shape[0] > K_1.shape[0]:
        resized_K_1 = resize_matrix(K_1, K_0.shape)
        combined_kernel = alpha * K_0 + (1 - alpha) * resized_K_1
    else:
        resized_K_0 = resize_matrix(K_0, K_1.shape)
        combined_kernel = alpha * resized_K_0 + (1 - alpha) * K_1

    return combined_kernel    

def to_csr_matrix(graph):
    """Convert a networkx graph to a CSR matrix."""
    return csr_matrix(nx.to_numpy_array(graph))

def generate_boundary_matrix_1(adj_matrix):
    """Generate the boundary matrix for 1-simplices (edges)."""
    num_vertices = adj_matrix.shape[0]
    edges = np.transpose(adj_matrix.nonzero())
    num_edges = edges.shape[0]
    B_1 = sp.lil_matrix((num_vertices, num_edges))
    for index, (i, j) in enumerate(edges):
        if i < j:
            B_1[i, index] = 1
            B_1[j, index] = -1
    return B_1.tocsr(), num_edges

def compute_hodge_laplacian_1(B_1):
    """Compute the Hodge Laplacian for 1-simplices."""
    return B_1.T @ B_1

def find_triangles(graph):
    """Find all triangles in the graph."""
    triangles = set()
    for node in graph.nodes():
        neighbors = set(graph.neighbors(node))
        for u in neighbors:
            for v in neighbors:
                if u != v and graph.has_edge(u, v) and node < u < v:
                    triangles.add((node, u, v))
    return list(triangles)

def generate_boundary_matrix_2(graph, triangles):
    """Generate the boundary matrix for 2-simplices (triangles)."""
    edge_map = {}
    edge_count = 0
    for u, v in graph.edges():
        sorted_edge = tuple(sorted((u, v)))
        if sorted_edge not in edge_map:
            edge_map[sorted_edge] = edge_count
            edge_count += 1

    B_2 = sp.lil_matrix((edge_count, len(triangles)))
    for i, triangle in enumerate(triangles):
        edges = [tuple(sorted((triangle[j], triangle[k]))) for j in range(3) for k in range(j + 1, 3)]
        for edge in edges:
            B_2[edge_map[edge], i] = 1
    return B_2.tocsr()

def find_tetrahedrons(graph):
    """Find all tetrahedrons in the graph."""
    tetrahedrons = set()
    for u in graph.nodes():
        neighbors_u = set(graph.neighbors(u))
        for v in neighbors_u:
            if u < v:
                common_uv = neighbors_u.intersection(set(graph.neighbors(v)))
                for w in common_uv:
                    if v < w:
                        common_uvw = common_uv.intersection(set(graph.neighbors(w)))
                        for x in common_uvw:
                            if w < x and graph.has_edge(u, x) and graph.has_edge(v, x) and graph.has_edge(w, x):
                                tetrahedrons.add((u, v, w, x))
    return list(tetrahedrons)

def is_tetrahedron(graph, nodes):
    """Check if the given nodes form a tetrahedron (complete graph K4) in the graph."""
    possible_edges = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]
    return all(graph.has_edge(*edge) for edge in possible_edges)

def generate_boundary_matrix_3(graph, triangles, tetrahedrons):
    """Generate the boundary matrix for 3-simplices (tetrahedrons)."""
    triangle_index = {tri: idx for idx, tri in enumerate(triangles)}
    B_3 = sp.lil_matrix((len(triangles), len(tetrahedrons)))
    for tet_idx, tet in enumerate(tetrahedrons):
        for trio in [(tet[i], tet[j], tet[k]) for i in range(4) for j in range(i + 1, 4) for k in range(j + 1, 4)]:
            tri = tuple(sorted(trio))
            if tri in triangle_index:
                B_3[triangle_index[tri], tet_idx] = 1 if (tet.index(tri[0]) + tet.index(tri[1]) + tet.index(tri[2])) % 2 == 0 else -1
    return B_3.tocsr()

def get_laplacian_0(graph):
    """Retrieve the Hodge Laplacian for 0-simplices (vertices) directly from the graph."""
    # Convert the graph to an adjacency matrix in CSR format
    A = nx.adjacency_matrix(graph)
    # Create the degree matrix
    degrees = np.array(A.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    # Compute the Laplacian
    L_0 = D - A
    return L_0
    
def get_laplacian_1(graph):
    """Retrieve the Hodge Laplacian for 1-simplices directly from the graph."""
    adj_matrix = to_csr_matrix(graph)
    B_1, _ = generate_boundary_matrix_1(adj_matrix)
    L_1 = compute_hodge_laplacian_1(B_1)
    return L_1

def get_laplacian_2(graph):
    """Retrieve the Hodge Laplacian for 2-simplices directly from the graph."""
    triangles = find_triangles(graph)
    B_2 = generate_boundary_matrix_2(graph, triangles)
    L_2 = B_2.T @ B_2
    return L_2

def get_laplacian_3(graph):
    """Retrieve the Hodge Laplacian for 3-simplices directly from the graph."""
    triangles = find_triangles(graph)
    tetrahedrons = find_tetrahedrons(graph)
    B_3 = generate_boundary_matrix_3(graph, triangles, tetrahedrons)
    L_3 = B_3.T @ B_3
    return L_3

def generate_simplices_edge(graph):
    simplices = []
    # Add edges (1-simplices)
    for u, v in graph.edges():
        simplices.append((min(u, v), max(u, v)))
    return simplices

def generate_simplices_triangle(graph):
    simplices = []
    # Add edges (1-simplices)
    for u, v in graph.edges():
        simplices.append((min(u, v), max(u, v)))

    # Add triangles (2-simplices)
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if graph.has_edge(neighbors[i], neighbors[j]):
                    triangle = tuple(sorted((node, neighbors[i], neighbors[j])))
                    simplices.append(triangle)
    return simplices

################################################

def generate_simplices(graph, dimension):
    simplices = []
    # Add nodes (0-simplices) if needed
    if dimension >= 0:
        for node in graph.nodes():
            simplices.append((node,))

    # Add edges (1-simplices)
    if dimension >= 1:
        for u, v in graph.edges():
            simplices.append((min(u, v), max(u, v)))

    # Higher-dimensional simplices
    if dimension >= 2:
        # Add triangles (2-simplices)
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if graph.has_edge(neighbors[i], neighbors[j]):
                        triangle = tuple(sorted((node, neighbors[i], neighbors[j])))
                        simplices.append(triangle)
        if dimension >= 3:
            # Add tetrahedrons (3-simplices)
            for node in graph.nodes():
                neighbors = list(graph.neighbors(node))
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        for k in range(j + 1, len(neighbors)):
                            if (graph.has_edge(neighbors[i], neighbors[j]) and 
                                graph.has_edge(neighbors[i], neighbors[k]) and 
                                graph.has_edge(neighbors[j], neighbors[k])):
                                tetrahedron = tuple(sorted((node, neighbors[i], neighbors[j], neighbors[k])))
                                simplices.append(tetrahedron)
    return simplices

def generate_simplices_tetrahedron(graph):
    simplices = []
    # Add edges (1-simplices)
    for u, v in graph.edges():
        simplices.append((min(u, v), max(u, v)))

    # Add triangles (2-simplices)
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if graph.has_edge(neighbors[i], neighbors[j]):
                    triangle = tuple(sorted((node, neighbors[i], neighbors[j])))
                    simplices.append(triangle)

    # Add tetrahedrons (3-simplices)
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    if (graph.has_edge(neighbors[i], neighbors[j]) and 
                        graph.has_edge(neighbors[i], neighbors[k]) and 
                        graph.has_edge(neighbors[j], neighbors[k])):
                        tetrahedron = tuple(sorted((node, neighbors[i], neighbors[j], neighbors[k])))
                        simplices.append(tetrahedron)

    return simplices

def transform_features(candidates, eig_vect):
    """Apply transformation using eigenvectors to candidate sets. Adjust for dimension discrepancies."""
    transformed = []
    eig_dim = eig_vect.shape[0]  # Dimensionality of the eigenvector space

    for candidate in candidates:
        # Convert candidate to numpy array if it's not already
        candidate_array = np.asarray(candidate)
        
        # Check if the candidate array matches the expected dimensions
        if candidate_array.size != eig_dim:
            # Handle dimension mismatch (could be error handling or padding)
            if candidate_array.size > eig_dim:
                # Truncate array if too long (not recommended without specific reason)
                candidate_array = candidate_array[:eig_dim]
            else:
                # Pad array with zeros if too short
                candidate_array = np.pad(candidate_array, (0, eig_dim - candidate_array.size), mode='constant')
        
        candidate_vector = candidate_array.reshape(-1, 1)
        transformed_vector = np.dot(eig_vect.T, candidate_vector).flatten()
        transformed.append(transformed_vector)
    return transformed

def normalize_laplacian_dense(L):
    """Normalize the Laplacian matrix."""
    D = np.diag(np.sqrt(1 / np.diag(L)))
    return np.dot(np.dot(D, L), D)

def normalize_laplacian_sparse(L):

    # Sum up the rows of L, get the degree of each vertex/simplice
    degree = np.array(L.sum(axis=1)).flatten()  # Convert to 1D numpy array if it's not already

    # Avoid division by zero in case of isolated nodes or similar issues
    degree[degree == 0] = 1

    # Create the diagonal matrix D^-1/2
    D_inv_sqrt = diags(1.0 / np.sqrt(degree))

    # Compute the normalized Laplacian L_sym = D^-1/2 * L * D^-1/2
    L_normalized = D_inv_sqrt.dot(L).dot(D_inv_sqrt)

    # Optionally convert to a dense matrix if needed and manageable size-wise
    # L_normalized_dense = L_normalized.toarray()  # Use this only if the matrix is not too large

    return L_normalized

def get_graph_infomation(G):

    # Check if the graph is directed
    is_directed = G.is_directed()

    # Calculate the density of the graph
    density = nx.density(G)

    # Check the number of connected components
    if is_directed:
        connected_components = nx.number_strongly_connected_components(G)
    else:
        connected_components = nx.number_connected_components(G)

    # Get the degree distribution
    degrees = [degree for node, degree in G.degree()]

    # Calculate the average clustering coefficient
    avg_clustering = nx.average_clustering(G)

    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    print("Directed:", is_directed)
    print("Density:", density)
    print("Connected Components:", connected_components)
    print("Average Degree:", sum(degrees) / len(degrees))
    print("Average Clustering Coefficient:", avg_clustering)

def estimate_max_simplex_degree(graph):
    # Convert the graph to a NetworkX graph if it isn't one already
    if not isinstance(graph, nx.Graph):
        G = nx.Graph(graph)
    else:
        G = graph

    # Find all cliques in the graph
    cliques = list(nx.find_cliques(G))
    # Find the maximum size of these cliques
    max_clique_size = max(len(clique) for clique in cliques)
    
    # The highest degree of simplices corresponds to the size of the largest clique minus one
    return max_clique_size - 1

def get_n_n_from_hodge(L0, L1, B):
    return L0.dot(B.dot(L1).dot(B.T))

def get_m_m_from_hodge(L0, L1, B):
    return L1.dot(B.T.dot(L0).dot(B))

####################### BOIM full helper begins #########################
def compute_eigen_decomposition(matrix):
    # Check if the input matrix is a sparse matrix
    if issparse(matrix):
        print("Sparse matrix detected.")
        # Check if it's a csr_matrix
        if isinstance(matrix, csr_matrix):
            print("CSR matrix detected.")
            # Convert csr_matrix to dense array using toarray()
            matrix_dense = matrix.toarray()
        elif isinstance(matrix, csc_matrix):
            print("CSC matrix detected.")
            # Convert csc_matrix to dense array using toarray()
            matrix_dense = matrix.toarray()
        else:
            print("Other sparse matrix format detected.")
            # For other sparse formats, use todense()
            matrix_dense = matrix.todense()
    else:
        print("Dense matrix detected.")
        # If it's already a dense matrix, use it directly
        matrix_dense = matrix

    # Compute eigenvalues and eigenvectors
    _, eig_vect = np.linalg.eigh(matrix_dense)

    # Compute the inverse of eigenvectors
    UT = np.linalg.inv(eig_vect)
    UT_inv = eig_vect
    # print("UT", UT)

    return UT, UT_inv

def perform_clustering_and_select_signals(number_of_clusters, sets_after_fourier_transfer):

    # print("candidate_sets", candidate_sets)
    # sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidate_sets, UT)
    # print("sets_after_fourier_transfer", sets_after_fourier_transfer)
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_

    groups = [[] for i in range(number_of_clusters)]
    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])
    
    # return groups, sets_after_fourier_transfer
    return groups

def simulate_initial_training(groups, G, config, num_of_sims, diffusion_model, number_of_sources, UT_inv):
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

    return torch.stack(train_X), torch.tensor(train_Y)

def iterative_optimization(train_X, train_Y, groups, number_of_sources, UT_inv, G, config, num_of_sims, diffusion_model, num_iterations):
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

    return train_X, train_Y, function_values, acquisition_values, max_train_Y_values, model

def identify_best_signal(sets_after_fourier_transfer, model, number_of_sources, UT_inv):
    best = float('-inf')
    identified_signal = None

    for signal in sets_after_fourier_transfer:
        input_tensor = torch.FloatTensor([signal])
        y_pred = model(input_tensor).loc
        if y_pred.item() > best:
            best = y_pred.item()
            identified_signal = signal

    identified_set = find_source_set_from_fourier(identified_signal, number_of_sources, UT_inv)
    return identified_set

####################### BOIM full helper ends #########################

####################### TDA helper begins #########################
import itertools

# Mapping Node Pairs to Edge Indices
def create_edge_index_mapping(G):
    edge_index = {}
    for idx, (u, v) in enumerate(G.edges()):
        # Ensure the tuple is ordered so the mapping is consistent
        edge_index[tuple(sorted((u, v)))] = idx
    return edge_index

# Generate Candidate Sets with Edge Indices
def create_candidate_set_pool_filtering_mapping(G, candidate_size, number_of_sources, allowed_shortest_distance):
    edge_index = create_edge_index_mapping(G)
    edges = list(G.edges())

    # Selecting top candidate edges based on some criteria, here simply random for demonstration
    random.shuffle(edges)
    candidate_edges = edges[:candidate_size]

    # Convert node pairs to edge indices
    candidate_edge_indices = [edge_index[tuple(sorted(edge))] for edge in candidate_edges]

    candidate_source_sets = []
    for selected_set in itertools.combinations(candidate_edge_indices, number_of_sources):
        # Here, implement any necessary logic to check distances or other criteria between edges
        candidate_source_sets.append(selected_set)

    # print('candidate source sets:', candidate_source_sets)
    return candidate_source_sets

def create_candidate_set_pool_filtering_mapping_centrality(G, candidate_size, number_of_sources, allowed_shortest_distance):
    edge_index = create_edge_index_mapping(G)

    # Select edges based on edge betweenness centrality
    centrality = nx.edge_betweenness_centrality(G)
    # Sort edges by centrality and pick the top candidates
    sorted_edges = sorted(G.edges(), key=lambda e: centrality[e], reverse=True)
    candidate_edges = sorted_edges[:candidate_size]

    candidate_edge_indices = [edge_index[tuple(sorted(edge))] for edge in candidate_edges]

    candidate_source_sets = []
    for selected_set in itertools.combinations(candidate_edge_indices, number_of_sources):
        # Implement any necessary logic to check distances or other criteria between edges
        candidate_source_sets.append(selected_set)

    return candidate_source_sets

def create_candidate_set_pool_filtering_mapping_cluster(G, candidate_size, number_of_sources, allowed_shortest_distance):
    edge_index = create_edge_index_mapping(G)
    
    # Calculate local clustering coefficient for each node
    clustering = nx.clustering(G)
    
    # Score edges by averaging the clustering coefficients of the nodes they connect
    scored_edges = {edge: (clustering[edge[0]] + clustering[edge[1]]) / 2 for edge in G.edges()}
    
    # Sort edges based on the clustering score and pick the top candidates
    sorted_edges = sorted(scored_edges, key=scored_edges.get, reverse=True)
    candidate_edges = [edge for edge in sorted_edges[:candidate_size]]
    
    candidate_edge_indices = [edge_index[tuple(sorted(edge))] for edge in candidate_edges]
    
    candidate_source_sets = []
    for selected_set in itertools.combinations(candidate_edge_indices, number_of_sources):
        # Implement any necessary logic to check distances or other criteria between edges
        edges_ok = True
        edge_pairs = itertools.combinations(selected_set, 2)
        for edge_a, edge_b in edge_pairs:
            nodes_a = list(G.edges())[edge_a][:2]  # Nodes of the first edge
            nodes_b = list(G.edges())[edge_b][:2]  # Nodes of the second edge
            # Check the shortest path distance between any nodes of the two edges
            distances = [nx.shortest_path_length(G, source=a, target=b) for a in nodes_a for b in nodes_b]
            if min(distances) <= allowed_shortest_distance:
                edges_ok = False
                break
        if edges_ok:
            candidate_source_sets.append(selected_set)

    return candidate_source_sets

def create_candidate_set_pool_filtering_mapping_centrality_cluster(G, candidate_size, number_of_sources, allowed_shortest_distance):
    edge_index = create_edge_index_mapping(G)

    # Calculate edge betweenness centrality and edge clustering coefficients
    centrality = nx.edge_betweenness_centrality(G)
    clustering = nx.clustering(G)

    # Score edges by combining centrality and clustering data
    scored_edges = {e: centrality[e] * clustering[e[0]] * clustering[e[1]] for e in G.edges()}

    # Sort edges by the combined score and pick the top candidates
    sorted_edges = sorted(scored_edges, key=scored_edges.get, reverse=True)
    candidate_edges = sorted_edges[:candidate_size]

    # Convert node pairs to edge indices
    candidate_edge_indices = [edge_index[tuple(sorted(edge))] for edge in candidate_edges]

    candidate_source_sets = []
    for selected_set in itertools.combinations(candidate_edge_indices, number_of_sources):
        edges_ok = True
        edge_pairs = itertools.combinations(selected_set, 2)
        for edge_a, edge_b in edge_pairs:
            edge_a_nodes = tuple(G.edges())[edge_a]
            edge_b_nodes = tuple(G.edges())[edge_b]
            # Check the shortest path distance between any nodes of the two edges
            distances = [nx.shortest_path_length(G, source=a, target=b) for a in edge_a_nodes for b in edge_b_nodes]
            if min(distances) <= allowed_shortest_distance:
                edges_ok = False
                break
        if edges_ok:
            candidate_source_sets.append(selected_set)

    return candidate_source_sets

def create_candidate_set_pool_filtering_mapping_weighted_edges(G, candidate_size, number_of_sources, allowed_shortest_distance):
    edge_index = create_edge_index_mapping(G)
    
    # Assume edges have a 'weight' attribute; correctly get edge data as a tuple of (node1, node2, attr_dict)
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    
    # Make sure to unpack the edge tuple to just get the edge (node1, node2) part without the data dictionary
    candidate_edges = [(u, v) for u, v, data in sorted_edges[:candidate_size]]
    
    candidate_edge_indices = [edge_index[tuple(sorted(edge))] for edge in candidate_edges]
    
    candidate_source_sets = []
    for selected_set in itertools.combinations(candidate_edge_indices, number_of_sources):
        # Implement logic to check distances or other criteria between edges
        candidate_source_sets.append(selected_set)
    
    # check_candidate_set(candidate_source_sets)  
    
    return candidate_source_sets

def create_candidate_set_pool_filtering_mapping_weighted_centrality(G, candidate_size, number_of_sources, allowed_shortest_distance):
    edge_index = create_edge_index_mapping(G)
    
    # Calculate edge centrality and assume edges have a 'weight' attribute
    centrality = nx.edge_betweenness_centrality(G)
    
    # Combine centrality and weight for sorting
    sorted_edges = sorted(G.edges(data=True), key=lambda x: (centrality[(x[0], x[1])], x[2]['weight']), reverse=True)
    
    # Extract the top candidate edges based on combined centrality and weight
    candidate_edges = [(u, v) for u, v, data in sorted_edges[:candidate_size]]
    
    candidate_edge_indices = [edge_index[tuple(sorted(edge))] for edge in candidate_edges]
    
    candidate_source_sets = []
    for selected_set in itertools.combinations(candidate_edge_indices, number_of_sources):
        # Implement logic to check distances or other criteria between edges
        candidate_source_sets.append(selected_set)
    
    return candidate_source_sets

def create_candidate_set_pool_filtering_mapping_weighted_cluster(G, candidate_size, number_of_sources, allowed_shortest_distance):
    edge_index = create_edge_index_mapping(G)
    
    # Calculate local clustering coefficient for each node
    clustering = nx.clustering(G)
    
    # Sort edges based on the product of edge weight and average clustering coefficient of the nodes forming the edge
    sorted_edges = sorted(G.edges(data=True), key=lambda x: (x[2]['weight'] * (clustering[x[0]] + clustering[x[1]]) / 2), reverse=True)
    
    # Select the top candidate edges based on the combined metric
    candidate_edges = [(u, v) for u, v, data in sorted_edges[:candidate_size]]
    
    candidate_edge_indices = [edge_index[tuple(sorted(edge))] for edge in candidate_edges]
    
    candidate_source_sets = []
    for selected_set in itertools.combinations(candidate_edge_indices, number_of_sources):
        # Implement any necessary logic to check distances or other criteria between edges
        candidate_source_sets.append(selected_set)
    
    return candidate_source_sets


# Adjust Fourier Transform for Edge Signals
# TODO this might be the same with fourier_transfer_for_all_candidate
# def fourier_transfer_for_all_candidate_set_mapping(candidate_sets, UT):
#     n = UT.shape[0]  # Assuming UT is square and matches the number of edges
#     signals = []

#     for source_set in candidate_sets:
#         a = np.zeros(n)
#         for edge_idx in source_set:
#             a[edge_idx] = 1  # Activate the signal at the edge index
#         signal = np.matmul(a, UT)
#         signals.append(signal)

#     return signals

# TODO this might be the smae with perform_clustering_and_select_signals
# def perform_clustering_and_select_signals_mapping(candidate_sets, UT, number_of_clusters):
# def perform_clustering_and_select_signals_mapping(number_of_clusters, sets_after_fourier_transfer):
#     # print("candidate_sets", candidate_sets)
#     # sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set_mapping(candidate_sets, UT)
#     # print("sets_after_fourier_transfer", sets_after_fourier_transfer)
#     kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
#     labels = kmeans.labels_

#     groups = [[] for i in range(number_of_clusters)]
#     for j in range(len(labels)):
#         groups[labels[j]].append(sets_after_fourier_transfer[j])
    
#     return groups, sets_after_fourier_transfer

from sklearn.decomposition import PCA
def reduce_features(features, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(features)
####################### TDA helper ends #########################

####################### Vanilla helper begins #########################
def compute_spectral_components(L, num_vectors=1):
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigsh(L, k=num_vectors + 1, which='SM', tol=1e-4)
    return eigenvalues, eigenvectors

# A 1180
def enhance_candidate_selection_with_spectra(G, candidates, eigenvectors, influence='fiedler'):
    # Use the Fiedler vector (2nd smallest eigenvalue) to rank nodes
    fiedler_vector = eigenvectors[:, 1]  # Second column corresponds to the second smallest eigenvalue
    node_rankings = {node: abs(fiedler_vector[i]) for i, node in enumerate(G.nodes())}
    # Sort candidates based on their spectral ranking
    sorted_candidates = sorted(candidates, key=lambda x: node_rankings[x], reverse=True)
    return sorted_candidates

# B 1190
def enhance_candidate_selection_with_centrality(G, candidates):
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)

    # Combine centrality measures with degree
    combined_scores = {}
    for node in candidates:
        combined_scores[node] = G.degree[node] + betweenness[node] + closeness[node]

    # Sort candidates by combined score
    sorted_candidates = sorted(combined_scores, key=combined_scores.get, reverse=True)
    
    return sorted_candidates

import networkx as nx

def enhance_candidate_selection_with_centrality_normalized(G, candidates):
    # Calculate centrality measures
    degree = dict(G.degree)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    
    # Normalize centrality measures
    max_degree = max(degree.values())
    max_betweenness = max(betweenness.values())
    max_closeness = max(closeness.values())
    
    normalized_degree = {node: degree[node] / max_degree for node in degree}
    normalized_betweenness = {node: betweenness[node] / max_betweenness for node in betweenness}
    normalized_closeness = {node: closeness[node] / max_closeness for node in closeness}

    # Weights for each centrality measure
    weight_degree = 1.0
    weight_betweenness = 1.0
    weight_closeness = 1.0
    
    # Combine centrality measures with weights
    combined_scores = {}
    for node in candidates:
        combined_scores[node] = (
            weight_degree * normalized_degree[node] +
            weight_betweenness * normalized_betweenness[node] +
            weight_closeness * normalized_closeness[node]
        )

    # Sort candidates by combined score
    sorted_candidates = sorted(combined_scores, key=combined_scores.get, reverse=True)
    
    return sorted_candidates

def enhance_candidate_selection_with_degree_betweenness_closeness_eigenvector(G, candidates):
    degree = dict(G.degree)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    eigenvector = nx.eigenvector_centrality(G)

    # Normalize centrality measures
    max_degree = max(degree.values())
    max_betweenness = max(betweenness.values())
    max_closeness = max(closeness.values())
    max_eigenvector = max(eigenvector.values())

    normalized_degree = {node: degree[node] / max_degree for node in degree}
    normalized_betweenness = {node: betweenness[node] / max_betweenness for node in betweenness}
    normalized_closeness = {node: closeness[node] / max_closeness for node in closeness}
    normalized_eigenvector = {node: eigenvector[node] / max_eigenvector for node in eigenvector}

    # Weights for each centrality measure
    weight_degree = 1.0
    weight_betweenness = 1.0
    weight_closeness = 1.0
    weight_eigenvector = 1.0

    # Combine centrality measures with weights
    combined_scores = {}
    for node in candidates:
        combined_scores[node] = (
            weight_degree * normalized_degree[node] +
            weight_betweenness * normalized_betweenness[node] +
            weight_closeness * normalized_closeness[node] +
            weight_eigenvector * normalized_eigenvector[node]
        )

    # Sort candidates by combined score
    sorted_candidates = sorted(combined_scores, key=combined_scores.get, reverse=True)
    
    return sorted_candidates

####################### Vanilla helper ends #########################

def evaluate_source(source_set, G, config, num_of_sims, diffusion_model):
    if diffusion_model == 'ic':
        e, _ = effectIC(G, config, source_set, num_of_sims)
    elif diffusion_model == 'lt':
        e, _ = effectLT(G, config, source_set, num_of_sims)
    else:
        raise NotImplementedError("Diffusion model not recognized.")
    return e

##################################
    # diffusion models
def IC(g, config, seed, rounds=100):
    result = []

    for iter in range(rounds):

        model_temp = ep.IndependentCascadesModel(g) # _temp
        config_temp = mc.Configuration()
        config_temp.add_model_initial_configuration('Infected', seed)

        for a, b in g.edges(): # _temp
            weight = config.config["edges"]['threshold'][(a, b)]
            # g_temp[a][b]['weight'] = weight
            config_temp.add_edge_configuration('threshold', (a, b), weight)

        model_temp.set_initial_status(config_temp)

        iterations = model_temp.iteration_bunch(5)

        total_no = 0

        for j in range(5):
            a = iterations[j]['node_count'][1]
            total_no += a

        result.append(total_no)

    return result

def LT(g, config, seed, rounds=100):
    result = []

    for iter in range(rounds):

        model_temp = ep.ThresholdModel(g) # _temp
        config_temp = mc.Configuration()
        config_temp.add_model_initial_configuration('Infected', seed)

        for a, b in g.edges(): # _temp
            weight = config.config["edges"]['threshold'][(a, b)]
            # g_temp[a][b]['weight'] = weight
            config_temp.add_edge_configuration('threshold', (a, b), weight)

        for i in g.nodes():
            threshold = random.randrange(1, 20)
            threshold = round(threshold / 100, 2)
            config_temp.add_node_configuration("threshold", i, threshold)

        model_temp.set_initial_status(config_temp)

        iterations = model_temp.iteration_bunch(5)

        total_no = iterations[4]['node_count'][1]

        result.append(total_no)

    return result

# Zonghan's code
def SI(g, config, seeds, rounds=100, beta=0.1):

    result = []

    for iter in range(rounds):

        model_temp = ep.SIModel(g) # _temp
        config_temp = mc.Configuration()
        config_temp.add_model_initial_configuration('Infected', seeds)
        config_temp.add_model_parameter('beta', beta)

        for a, b in g.edges(): # _temp
            weight = config.config["edges"]['threshold'][(a, b)]
            config_temp.add_edge_configuration('threshold', (a, b), weight)

        model_temp.set_initial_status(config_temp)

        iterations = model_temp.iteration_bunch(5)

        result.append(iterations[4]['node_count'][1])

    return result

#### IMM ####
def sampling_imm(epsoid, l, graph, node_num, seed_size, model):
    R = []
    LB = 1
    n = node_num
    k = seed_size
    epsoid_p = epsoid * math.sqrt(2)

    for i in range(1, int(math.log2(n-1))+1):
        s = time.time()
        x = n/(math.pow(2, i))
        lambda_p = ((2+2*epsoid_p/3)*(logcnk(n, k) + l*math.log(n) + math.log(math.log2(n)))*n)/pow(epsoid_p, 2)
        theta = lambda_p/x

        for _ in range(int(theta) - len(R)):
            v = random.randint(0, node_num - 1)
            rr = generate_rr(v, graph, node_num, model)
            R.append(rr)

        end = time.time()
        print('time to find rr', end - s)
        start = time.time()
        Si, f = node_selection(R, k, node_num)
        print(f)
        end = time.time()
        print('node selection time', time.time() - start)

        if n * f >= (1 + epsoid_p) * x:
            LB = n * f / (1 + epsoid_p)
            break

    alpha = math.sqrt(l * math.log(n) + math.log(2))
    beta = math.sqrt((1 - 1 / math.e) * (logcnk(n, k) + l * math.log(n) + math.log(2)))
    lambda_aster = 2 * n * pow(((1 - 1 / math.e) * alpha + beta), 2) * pow(epsoid, -2)
    theta = lambda_aster / LB
    length_r = len(R)
    diff = int(theta - length_r)

    if diff > 0:
        for _ in range(diff):
            v = random.randint(0, node_num - 1)
            rr = generate_rr(v, graph, node_num, model)
            R.append(rr)

    return R

def generate_rr(v, graph, node_num, model):
    if model == 'IC':
        return generate_rr_ic(v, graph)
    elif model == 'LT':
        return generate_rr_lt(v, graph)
    elif model == 'SI':
        return generate_rr_si(v, graph)

def node_selection(R, k, node_num):
    Sk = []
    rr_degree = [0 for _ in range(node_num)]
    node_rr_set = dict()
    matched_count = 0

    for j in range(len(R)):
        rr = R[j]
        for rr_node in rr:
            rr_degree[rr_node] += 1
            if rr_node not in node_rr_set:
                node_rr_set[rr_node] = list()
            node_rr_set[rr_node].append(j)

    for _ in range(k):
        max_point = rr_degree.index(max(rr_degree))
        Sk.append(max_point)
        matched_count += len(node_rr_set[max_point])
        index_set = list(node_rr_set[max_point])
        for jj in index_set:
            rr = R[jj]
            for rr_node in rr:
                rr_degree[rr_node] -= 1
                node_rr_set[rr_node].remove(jj)

    return Sk, matched_count / len(R)

def generate_rr_ic(node, graph):
    activity_set = [node]
    activity_nodes = [node]

    while activity_set:
        new_activity_set = []
        for seed in activity_set:
            for neighbor in graph.neighbors(seed):
                weight = graph.edges[seed, neighbor].get('weight', 1.0)
                if neighbor not in activity_nodes and random.random() < weight:
                    activity_nodes.append(neighbor)
                    new_activity_set.append(neighbor)
        activity_set = new_activity_set

    return activity_nodes

def generate_rr_lt(node, graph):
    activity_nodes = [node]
    activity_set = node

    while activity_set != -1:
        new_activity_set = -1
        neighbors = list(graph.neighbors(activity_set))
        if len(neighbors) == 0:
            break
        candidate = random.sample(neighbors, 1)[0]
        if candidate not in activity_nodes:
            activity_nodes.append(candidate)
            new_activity_set = candidate
        activity_set = new_activity_set

    return activity_nodes

def generate_rr_si(node, graph):
    activity_set = [node]
    activity_nodes = [node]

    while activity_set:
        new_activity_set = []
        for seed in activity_set:
            for neighbor in graph.neighbors(seed):
                if neighbor not in activity_nodes:
                    activity_nodes.append(neighbor)
                    new_activity_set.append(neighbor)
        activity_set = new_activity_set

    return activity_nodes

def logcnk(n, k):
    res = 0
    for i in range(n - k + 1, n + 1):
        res += math.log(i)
    for i in range(1, k + 1):
        res -= math.log(i)
    return res

######## RIS        

def get_RRS(g, config):
    """
    Inputs: g: Network graph
            config: Configuration object for the IC model
    Outputs: A random reverse reachable set expressed as a list of nodes
    """
    # get edges according to the propagation probability
    edges = [(u, v) for (u, v, d) in g.edges(data=True) if uniform(0, 1) < config.config["edges"]['threshold'][(u, v)]]
    
    # create a subgraph based on the edges
    g_sub = g.edge_subgraph(edges)
    
    # select a random node as the starting point that = part of the subgraph
    source = random.choice(list(g_sub.nodes()))
    
    # perform a depth-first traversal from the source node to get the RRS
    RRS = list(nx.dfs_preorder_nodes(g_sub, source))
    return RRS

######## IMRank
def LFA(matrix):
    """
    Linear Feedback Algorithm to update the ranks of the nodes.
    """
    n = len(matrix)
    Mr = [1 for _ in range(n)]
    Mr_next = Mr.copy()
    for i_ in range(1, n):
        i = n - i_
        for j in range(0, i + 1):
            Mr_next[j] = Mr_next[j] + matrix[j][i] * Mr[i]
            Mr_next[i] = (1 - matrix[j][i]) * Mr_next[i]
        Mr = Mr_next.copy()
    return Mr


#### NNGP
class GNNGPBoTorchWrapper(Model):
    def __init__(self, gnngp_model, epsilon):
        super().__init__()
        self.gnngp_model = gnngp_model
        self.epsilon = epsilon

    @property
    def num_outputs(self):
        return 1

    def posterior(self, X, observation_noise=False, **kwargs):
        # Generate predictions using the GNNGP model
        with torch.no_grad():
            fit_result = self.gnngp_model.predict(self.epsilon)
            mean = fit_result.mean(dim=1)  # Ensure mean is correct shape
            variance = torch.var(fit_result, dim=1)  # Estimate variance as the variance of predictions
            covar = torch.diag_embed(variance)  # Create a square matrix with variance on the diagonal

            if covar.dim() == 2:
                covar = covar.unsqueeze(0)  # Ensure covar is at least 2D
            
            mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=covar)
            return GPyTorchPosterior(mvn)

class NTKBoTorchWrapper(Model):
    def __init__(self, ntk_kernel, epsilon):
        super().__init__()
        self.ntk_kernel = torch.tensor(ntk_kernel, dtype=torch.float32)  # Convert NTK kernel to Tensor
        self.epsilon = epsilon

    @property
    def num_outputs(self):
        return 1

    def posterior(self, X, observation_noise=False, **kwargs):
        with torch.no_grad():
            X_np = X.detach().cpu().numpy()
            ntk_kernel_tensor = self.ntk_kernel

            # print(f"X_np shape: {X_np.shape}")
            # print(f"ntk_kernel_tensor shape: {ntk_kernel_tensor.shape}")

            # Reshape X_np to 2D if it's 3D
            if X_np.ndim == 3:
                X_np = X_np.squeeze(1)  # Remove the middle dimension if it's 1
            elif X_np.ndim != 2:
                raise ValueError(f"Unexpected input shape: {X_np.shape}")

            # Adjust X_np or ntk_kernel_tensor to match sizes
            if X_np.shape[1] != ntk_kernel_tensor.shape[0]:
                # print(f"Adjusting shapes: X_np {X_np.shape}, kernel {ntk_kernel_tensor.shape}")
                if X_np.shape[1] < ntk_kernel_tensor.shape[0]:
                    # Pad X_np with zeros
                    pad_width = ((0, 0), (0, ntk_kernel_tensor.shape[0] - X_np.shape[1]))
                    X_np = np.pad(X_np, pad_width, mode='constant')
                else:
                    # Truncate X_np
                    X_np = X_np[:, :ntk_kernel_tensor.shape[0]]

            # Compute mean and covariance
            mean = torch.tensor(X_np @ ntk_kernel_tensor.numpy(), dtype=torch.float32)
            variance = torch.var(mean, dim=0)
            covar = torch.diag_embed(variance) + torch.eye(variance.size(0)) * 1e-6  # Adding jitter

            if covar.dim() == 2:
                covar = covar.unsqueeze(0)

            mvn = torch.distributions.MultivariateNormal(mean, covariance_matrix=covar)
            return GPyTorchPosterior(mvn)

def psd_safe_cholesky(A, upper=False, jitter=1e-8, max_tries=10):
    L = None
    for i in range(max_tries):
        try:
            matrix = A.clone()
            matrix.diagonal().add_(jitter)
            L = torch.linalg.cholesky(matrix, upper=upper)
            return L
        except RuntimeError:
            jitter *= 10
    raise RuntimeError("Matrix not positive definite even after adding jitter")
