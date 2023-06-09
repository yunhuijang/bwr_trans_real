import networkx as nx
import numpy as np
import scipy.stats
from graph_gen.analysis import graph_metrics
from graph_gen.data.bandwidth import BandFlatten, graph_from_bandflat

def make_histograms(
	value_arrs, num_bins=None, bin_width=None, bin_array=None, frequency=False,
	epsilon=1e-6
):
	"""
	Given a set of value arrays, converts them into histograms of counts or
	frequencies. The bins may be specified as a number of bins, a bin width, or
	a pre-defined array of bin edges. This function creates histograms such that
	all value arrays given are transformed into the same histogram space.
	Arguments:
		`value_arrs`: an iterable of N 1D NumPy arrays to make histograms of
		`num_bins`: if given, make the histograms have this number of bins total
		`bin_width`: if given, make the histograms have bins of this width,
			aligned starting at the minimum value
		`bin_array`: if given, make the histograms according to this NumPy array
			of bin edges
		`frequency`: if True, normalize each histogram into frequencies
		`epsilon`: small number for stability of last endpoint if `bin_width` is
			specified
	Returns an N x B array of counts or frequencies (N is parallel to the input
	`value_arrs`), where B is the number of bins in the histograms.
	"""
	# Compute bins if needed
	if num_bins is not None:
		assert bin_width is None and bin_array is None
		min_val = min(np.nanmin(arr) for arr in value_arrs)
		max_val = max(np.nanmax(arr) for arr in value_arrs)
		bin_array = np.linspace(min_val, max_val, num_bins + 1)
	elif bin_width is not None:
		assert num_bins is None and bin_array is None
		min_val = min(np.nanmin(arr) for arr in value_arrs)
		max_val = max(np.nanmax(arr) for arr in value_arrs) + bin_width + \
			epsilon
		bin_array = np.arange(min_val, max_val, bin_width)
	elif bin_array is not None:
		assert num_bins is None and bin_width is None
	else:
		raise ValueError(
			"Must specify one of `num_bins`, `bin_width`, or `bin_array`"
		)

	# Compute histograms
	hists = np.empty((len(value_arrs), len(bin_array) - 1))
	for i, arr in enumerate(value_arrs):
		hist = np.histogram(arr, bins=bin_array)[0]
		if frequency:
			hist = hist / len(arr)
		hists[i] = hist
	
	return hists


def gaussian_kernel(vec_1, vec_2, sigma=1):
	"""
	Computes the Gaussian kernel function on two vectors. This is also known as
	the radial basis function. 
	Arguments:
		`vec_1`: a NumPy array of values
		`vec_2`: a NumPy array of values; the underlying vector space must be
			the same as `vec_1`
		`sigma`: standard deviation for the Gaussian kernel
	Returns a scalar similarity value between 0 and 1.
	"""
	l2_dist_squared = np.sum(np.square(vec_1 - vec_2))
	return np.exp(-l2_dist_squared / (2 * sigma * sigma))


def gaussian_wasserstein_kernel(vec_1, vec_2, sigma=1):
	"""
	Computes the Gaussian kernel function on two vectors, where the similarity
	metric within the Gaussian is the Wasserstein distance (i.e. Earthmover's
	distance). The two vectors must be distributions represented as PMFs over
	the same probability space.
	Arguments:
		`vec_1`: a NumPy array representing a PMF distribution (values are
			probabilities)
		`vec_2`: a NumPy array representing a PMF distribution (values are
			probabilities); the underlying probability space (i.e. support) must
			be the same as `vec_1`
		`sigma`: standard deviation for the Gaussian kernel
	Returns a scalar similarity value between 0 and 1.
	"""
	assert vec_1.shape == vec_2.shape
	# Since the vectors are supposed to be PMFs, if everything is 0 then just
	# turn it into an (unnormalized) uniform distribution
	if np.all(vec_1 == 0):
		vec_1 = np.ones_like(vec_1)
	if np.all(vec_2 == 0):
		vec_2 = np.ones_like(vec_2)
	# The SciPy Wasserstein distance function takes in empirical observations
	# instead of histograms/distributions as an input, but we can get the same
	# result by specifying weights which are the PMF probabilities
	w_dist = scipy.stats.wasserstein_distance(
		np.arange(len(vec_1)), np.arange(len(vec_1)), vec_1, vec_2
	)
	return np.exp(-(w_dist * w_dist) / (2 * sigma * sigma))


def gaussian_total_variation_kernel(vec_1, vec_2, sigma=1):
	"""
	Computes the Gaussian kernel function on two vectors, where the similarity
	metric within the Gaussian is the total variation between the two vectors.
	Arguments:
		`vec_1`: a NumPy array of values
		`vec_2`: a NumPy array of values; the underlying vector space must be
			the same as `vec_1`
		`sigma`: standard deviation for the Gaussian kernel
	Returns a scalar similarity value between 0 and 1.
	"""
	tv_dist = np.sum(np.abs(vec_1 - vec_2)) / 2
	return np.exp(-(tv_dist * tv_dist) / (2 * sigma * sigma))


def compute_inner_prod_feature_mean(dist_1, dist_2, kernel_type, **kwargs):
	"""
	Given two empirical distributions of vectors, computes the inner product of
	the feature means using the specified kernel. This is equivalent to the
	expected/average kernel function on all pairs of vectors between the two
	distributions.
	Arguments:
		`dist_1`: an M x D NumPy array of M vectors, each of size D; all vectors
			must share the same underlying vector space (or probability space if
			the vectors represent a probability distribution) with each other
			and with `dist_2`
		`dist_2`: an M x D NumPy array of M vectors, each of size D; all vectors
			must share the same underlying vector space (or probability space if
			the vectors represent a probability distribution) with each other
			and with `dist_1`
		`kernel_type`: type of kernel to apply for computing the kernelized
			inner product; can be "gaussian", "gaussian_wasserstein", or
			"gaussian_total_variation"
		`kwargs`: extra keyword arguments to be passed to the kernel function
	Returns a scalar which is the average kernelized inner product between all
	pairs of vectors across the two distributions.
	"""
	if kernel_type == "gaussian":
		kernel_func = gaussian_kernel
	elif kernel_type == "gaussian_wasserstein":
		kernel_func = gaussian_wasserstein_kernel
	elif kernel_type == "gaussian_total_variation":
		kernel_func = gaussian_total_variation_kernel
	else:
		raise ValueError("Unknown kernel type: %s" % kernel_type)
	
	inner_prods = []
	for vec_1 in dist_1:
		for vec_2 in dist_2:
			inner_prods.append(kernel_func(vec_1, vec_2, **kwargs))
	
	return np.mean(inner_prods)


def compute_maximum_mean_discrepancy(
	dist_1, dist_2, kernel_type, normalize=True, **kwargs
):
	"""
	Given two empirical distributions of vectors, computes the maximum mean
	discrepancy (MMD) between the two distributions.
	Arguments:
		`dist_1`: an M x D NumPy array of M vectors, each of size D; all vectors
			must share the same underlying vector space (or probability space if
			the vectors represent a probability distribution) with each other
			and with `dist_2`
		`dist_2`: an M x D NumPy array of M vectors, each of size D; all vectors
			must share the same underlying vector space (or probability space if
			the vectors represent a probability distribution) with each other
			and with `dist_1`
		`kernel_type`: type of kernel to apply for computing the kernelized
			inner product; can be "gaussian", "gaussian_wasserstein", or
			"gaussian_total_variation"
		`normalize`: if True, normalize each D-vector to sum to 1
		`kwargs`: extra keyword arguments to be passed to the kernel function
	Returns the scalar MMD value.
	"""
	if normalize:
		dist_1 = dist_1 / np.sum(dist_1, axis=1, keepdims=True)
		dist_2 = dist_2 / np.sum(dist_2, axis=1, keepdims=True)

	term_1 = compute_inner_prod_feature_mean(
		dist_1, dist_1, kernel_type, **kwargs
	)
	term_2 = compute_inner_prod_feature_mean(
		dist_2, dist_2, kernel_type, **kwargs
	)
	term_3 = compute_inner_prod_feature_mean(
		dist_1, dist_2, kernel_type, **kwargs
	)
	return np.sqrt(term_1 + term_2 - (2 * term_3))


def get_graph_statistics(graphs: list) -> dict:
	if len(graphs) == 0:
		print('No graph')
		return {
		"degree": [np.array([0])],
		"cluster": [np.array([0])],
		"spectra": [np.array([0])],
		"orbit": [np.array([0])],
	}

	
	degrees = graph_metrics.get_degrees(graphs)
	cluster_coefs = graph_metrics.get_clustering_coefficients(graphs)
	spectra = graph_metrics.get_spectra(graphs)
	orbit_counts = graph_metrics.get_orbit_counts(graphs)
	orbit_counts = np.stack([
		np.mean(counts, axis=0) for counts in orbit_counts
	])
	return {
		"degree": degrees,
		"cluster": cluster_coefs,
		"spectra": spectra,
		"orbit": orbit_counts,
	}


def evaluate_sampled_graphs(sampled_graphs: list, real_graphs: list) -> dict:
	sampled_stats = get_graph_statistics(sampled_graphs)
	actual_stats = get_graph_statistics(real_graphs)
	out = {}
	kernel_type = "gaussian_total_variation"
	# degree
	max_deg = max(
		max(map(max, actual_stats["degree"])),
		max(map(max, sampled_stats["degree"])),
	)
	degree_bins = np.arange(0.5, max_deg + 0.5)
	actual_degree_hist = make_histograms(actual_stats["degree"], bin_array=degree_bins)
	sampled_degree_hist = make_histograms(sampled_stats["degree"], bin_array=degree_bins)
	out["degree_mmd"] = compute_maximum_mean_discrepancy(
		actual_degree_hist, sampled_degree_hist, kernel_type=kernel_type, sigma=1,
	)
	# cluster
	actual_cluster_hist = make_histograms(actual_stats["cluster"], num_bins=100)
	sampled_cluster_hist = make_histograms(sampled_stats["cluster"], num_bins=100)
	out["cluster_mmd"] = compute_maximum_mean_discrepancy(
		actual_cluster_hist, sampled_cluster_hist, kernel_type=kernel_type, sigma=0.1,
	)
	# cluster
	bin_array = np.linspace(1e-5, 2, 201)
	actual_spectra_hist = make_histograms(actual_stats["spectra"], bin_array=bin_array)
	sampled_spectra_hist = make_histograms(sampled_stats["spectra"], bin_array=bin_array)
	out["spectra_mmd"] = compute_maximum_mean_discrepancy(
		actual_spectra_hist, sampled_spectra_hist, kernel_type=kernel_type, sigma=1,
	)
	# orbit
	out["orbit_mmd"] = compute_maximum_mean_discrepancy(
		actual_stats["orbit"], sampled_stats["orbit"], kernel_type=kernel_type,
		normalize=False, sigma=30,
	)
	return out