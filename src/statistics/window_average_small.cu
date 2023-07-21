#include <device_launch_parameters.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>

#include "../diagnostics.h"
#include "../utils.h"
#include "window_average_small.h"
#include <fstream>
__device__ int get_non_internal_index(const state_t& s, const state_t& internal_mask)
{
	int idx = 0;
	int idx_i = 0;

	for (int i = 0; i < states_count; i++)
	{
		if (!internal_mask.is_set(i))
		{
			int multiplier = s.is_set(i) ? 1 : 0;

			idx += multiplier * (1 << idx_i);

			idx_i++;
		}
	}

	return idx;
}

__global__ void window_average_small(int max_traj_len, int n_trajectories, float window_size, state_t internal_mask,
									 int noninternal_states_count, const state_t* __restrict__ traj_states,
									 const float* __restrict__ traj_times, const float* __restrict__ traj_tr_entropies,
									 float* __restrict__ window_probs, float* __restrict__ window_tr_entropies)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= n_trajectories * max_traj_len)
		return;

	if (id % max_traj_len == 0 || traj_times[id] == 0.f)
	{
		return;
	}

	const int state_idx = get_non_internal_index(traj_states[id], internal_mask);
	const float tr_h = traj_tr_entropies[id];

	float slice_begin = traj_times[id - 1];
	float slice_end = traj_times[id];
	int wnd_idx = floorf(slice_begin / window_size);

	while (slice_end > slice_begin)
	{
		float wnd_end = (wnd_idx + 1) * window_size;
		float slice_in_wnd = fminf(slice_end, wnd_end) - slice_begin;

		atomicAdd(window_probs + (wnd_idx * noninternal_states_count + state_idx), slice_in_wnd);
		atomicAdd(window_tr_entropies + wnd_idx, tr_h * slice_in_wnd);

		wnd_idx++;
		slice_begin = fminf(slice_end, wnd_end);
	}
}

__global__ void window_average_small_discrete(int max_traj_len, int n_trajectories, float window_size,
											  state_t internal_mask, int noninternal_states_count,
											  const state_t* __restrict__ traj_states,
											  const float* __restrict__ traj_times,
											  const float* __restrict__ traj_tr_entropies,
											  int* __restrict__ window_probs, float* __restrict__ window_tr_entropies)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= n_trajectories * max_traj_len)
		return;

	if (id % max_traj_len == 0 || traj_times[id] == 0.f)
	{
		return;
	}

	const int state_idx = get_non_internal_index(traj_states[id], internal_mask);
	const float tr_h = traj_tr_entropies[id];

	int wnd_idx = lroundf(traj_times[id - 1] / window_size);

	atomicAdd(window_probs + (wnd_idx * noninternal_states_count + state_idx), 1);
	atomicAdd(window_tr_entropies + wnd_idx, tr_h);
}

window_average_small_stats::window_average_small_stats(float window_size, float max_time, bool discrete_time,
													   state_t internal_mask, size_t non_internals, size_t max_traj_len,
													   size_t max_n_trajectories)
	: window_size_(window_size),
	  max_time_(max_time),
	  discrete_time_(discrete_time),
	  internal_mask_(internal_mask),
	  noninternal_states_count_(1 << non_internals),
	  max_traj_len_(max_traj_len),
	  max_n_trajectories_(max_n_trajectories)
{
	timer t;
	t.start();

	size_t windows_count = std::ceil(max_time / window_size);

	window_tr_entropies_ = thrust::device_malloc<float>(windows_count);
	result_tr_entropies_.resize(windows_count);
	CUDA_CHECK(cudaMemset(window_tr_entropies_.get(), 0, windows_count * sizeof(float)));

	if (discrete_time)
	{
		window_probs_discrete_ = thrust::device_malloc<int>(windows_count * noninternal_states_count_);
		result_probs_discrete_.resize(windows_count * noninternal_states_count_);
		CUDA_CHECK(
			cudaMemset(window_probs_discrete_.get(), 0, windows_count * noninternal_states_count_ * sizeof(int)));
	}
	else
	{
		window_probs_ = thrust::device_malloc<float>(windows_count * noninternal_states_count_);
		result_probs_.resize(windows_count * noninternal_states_count_);
		CUDA_CHECK(cudaMemset(window_probs_.get(), 0, windows_count * noninternal_states_count_ * sizeof(float)));
	}

	CUDA_CHECK(cudaDeviceSynchronize());

	t.stop();

	if (print_diags)
	{
		std::cout << "window_average_small> init_time: " << t.millisecs() << "ms" << std::endl;
	}
}

window_average_small_stats::~window_average_small_stats()
{
	thrust::device_free(window_probs_);
	thrust::device_free(window_tr_entropies_);
}

void window_average_small_stats::process_batch(thrust::device_ptr<state_t> traj_states,
											   thrust::device_ptr<float> traj_times,
											   thrust::device_ptr<float> traj_tr_entropies, thrust::device_ptr<state_t>,
											   thrust::device_ptr<trajectory_status>, int n_trajectories)
{
	process_batch_internal(traj_states, traj_times, traj_tr_entropies, n_trajectories);
}

void window_average_small_stats::process_batch_internal(thrust::device_ptr<state_t> traj_states,
														thrust::device_ptr<float> traj_times,
														thrust::device_ptr<float> traj_tr_entropies, int n_trajectories)
{
	timer t;
	t.start();

	if (discrete_time_)
	{
		window_average_small_discrete<<<DIV_UP(max_traj_len_ * n_trajectories, 512), 512>>>(
			max_traj_len_, n_trajectories, window_size_, internal_mask_, noninternal_states_count_, traj_states.get(),
			traj_times.get(), traj_tr_entropies.get(), window_probs_discrete_.get(), window_tr_entropies_.get());
	}
	else
	{
		window_average_small<<<DIV_UP(max_traj_len_ * n_trajectories, 512), 512>>>(
			max_traj_len_, n_trajectories, window_size_, internal_mask_, noninternal_states_count_, traj_states.get(),
			traj_times.get(), traj_tr_entropies.get(), window_probs_.get(), window_tr_entropies_.get());
	}

	CUDA_CHECK(cudaDeviceSynchronize());

	t.stop();

	if (print_diags)
	{
		std::cout << "window_average_small> reduce_time: " << t.millisecs() << "ms" << std::endl;
	}
}

void window_average_small_stats::finalize()
{
	size_t windows_count = std::ceil(max_time_ / window_size_);
	timer t;

	t.start();

	// copy result data into host
	if (discrete_time_)
		CUDA_CHECK(cudaMemcpy(result_probs_discrete_.data(), thrust::raw_pointer_cast(window_probs_discrete_),
							  windows_count * noninternal_states_count_ * sizeof(int), cudaMemcpyDeviceToHost));
	else
		CUDA_CHECK(cudaMemcpy(result_probs_.data(), thrust::raw_pointer_cast(window_probs_),
							  windows_count * noninternal_states_count_ * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(result_tr_entropies_.data(), thrust::raw_pointer_cast(window_tr_entropies_),
						  windows_count * sizeof(float), cudaMemcpyDeviceToHost));

	t.stop();

	if (print_diags)
	{
		std::cout << "window_average_small> finalize_time: " << t.millisecs() << "ms" << std::endl;
	}
}

state_t non_internal_idx_to_state(const state_t& internal_mask, int idx)
{
	state_t ret;
	size_t idx_i = 0;
	for (size_t i = 0; i < states_count; i++)
	{
		if (!internal_mask.is_set(i))
		{
			if ((idx & (1 << idx_i)) != 0)
				ret.set(i);
			idx_i++;
		}
	}

	return ret;
}

float window_average_small_stats::get_single_result_prob(int n_trajectories, size_t idx)
{
	if (discrete_time_)
	{
		auto occurences = result_probs_discrete_[idx];

		return (float)occurences / (float)n_trajectories;
	}
	else
	{
		auto cumul_slices = result_probs_[idx];

		return cumul_slices / (n_trajectories * window_size_);
	}
}

void window_average_small_stats::visualize(int n_trajectories, const std::vector<std::string>& nodes)
{
	size_t windows_count = std::ceil(max_time_ / window_size_);

	for (size_t i = 0; i < windows_count; ++i)
	{
		float entropy = 0.f;
		float wnd_tr_entropy = result_tr_entropies_[i] / n_trajectories;
		wnd_tr_entropy /= discrete_time_ ? 1 : window_size_;

		for (size_t s_idx = 0; s_idx < noninternal_states_count_; s_idx++)
		{
			auto prob = get_single_result_prob(n_trajectories, i * noninternal_states_count_ + s_idx);

			if (prob == 0.f)
				continue;

			entropy += -std::log2(prob) * prob;
		}

		std::cout << "window (" << i * window_size_ << ", " << (i + 1) * window_size_ << "]" << std::endl;
		std::cout << "entropy: " << entropy << std::endl;
		std::cout << "transition entropy: " << wnd_tr_entropy << std::endl;

		for (size_t s_idx = 0; s_idx < noninternal_states_count_; s_idx++)
		{
			auto prob = get_single_result_prob(n_trajectories, i * noninternal_states_count_ + s_idx);

			if (prob == 0.f)
				continue;

			std::cout << prob << " " << to_string(non_internal_idx_to_state(internal_mask_, s_idx), nodes) << std::endl;
		}
	}
}


void window_average_small_stats::writeCSV(int n_trajectories, const std::vector<std::string>& nodes, const std::string prefix)
{
	size_t windows_count = std::ceil(max_time_ / window_size_);
	std::ofstream ofs;
	
	ofs.open(prefix + "_probtraj.csv");
	if (ofs) 
	{
		// Computing max states for header
		int max_states = 0;
		for (size_t i = 0; i < windows_count; ++i)
		{
			int num_states = 0;			
			for (size_t s_idx = 0; s_idx < noninternal_states_count_; s_idx++)
			{
				auto prob = get_single_result_prob(n_trajectories, i * noninternal_states_count_ + s_idx);

				if (prob == 0.f)
					continue;
					
				num_states += 1;
			}
			
			max_states = std::max(max_states, num_states);
		}
		
		// writing header
		ofs << "Time\tTH\tErrorTH\tH\tHD=0\t";
		for (int i = 0; i < max_states; i++)
		{
			ofs << "State\tProba\tErrorProba";
			if (i < max_states - 1) {
				ofs << "\t";
			}
		}
		
		for (size_t i = 0; i < windows_count; ++i)
		{
			float entropy = 0.f;
			float wnd_tr_entropy = result_tr_entropies_[i] / n_trajectories;
			wnd_tr_entropy /= discrete_time_ ? 1 : window_size_;

			for (size_t s_idx = 0; s_idx < noninternal_states_count_; s_idx++)
			{
				auto prob = get_single_result_prob(n_trajectories, i * noninternal_states_count_ + s_idx);

				if (prob == 0.f)
					continue;

				entropy += -std::log2(prob) * prob;
			}
			ofs << i * window_size_ << "\t";
			// std::cout << "window (" << i * window_size_ << ", " << (i + 1) * window_size_ << "]" << std::endl;
			// std::cout << "entropy: " << entropy << std::endl;
			// std::cout << "transition entropy: " << wnd_tr_entropy << std::endl;
			ofs << wnd_tr_entropy << "\t" << 0.f << "\t" << entropy << "\t" << 0.f << "\t";

			for (size_t s_idx = 0; s_idx < noninternal_states_count_; s_idx++)
			{
				auto prob = get_single_result_prob(n_trajectories, i * noninternal_states_count_ + s_idx);

				if (prob == 0.f)
					continue;

				std::cout << prob << " " << to_string(non_internal_idx_to_state(internal_mask_, s_idx), nodes) << std::endl;
				ofs << to_string(non_internal_idx_to_state(internal_mask_, s_idx), nodes) << "\t" << prob << "\t" << 0.f;
				if (s_idx < (noninternal_states_count_ - 1))
				{
					ofs << "\t";
				}
			}
			ofs << std::endl;
		}
	}
}
