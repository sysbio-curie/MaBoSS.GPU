#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/zip_function.h>

#include "../utils.h"
#include "window_average.h"

constexpr bool print_diags = false;

namespace thrust {
template <>
struct plus<tuple<float, float>>
{
	__host__ __device__ tuple<float, float> operator()(const tuple<float, float>& lhs,
													   const tuple<float, float>& rhs) const
	{
		return make_tuple(lhs.get<0>() + rhs.get<0>(), lhs.get<1>() + rhs.get<1>());
	}
};
} // namespace thrust

window_average_stats::window_average_stats() : batch_size_limit_(50'000'000) {}

thrust::device_vector<int> partition_count(float window_size, thrust::device_ptr<state_t> traj_states,
										   thrust::device_ptr<float> traj_times,
										   thrust::device_ptr<float> traj_tr_entropies, int max_traj_len,
										   int n_trajectories)
{
	thrust::device_vector<int> traj_step_in_windows(n_trajectories * max_traj_len + 1);

	thrust::for_each_n(
		thrust::device, thrust::make_counting_iterator(0), n_trajectories * max_traj_len,
		[max_traj_len, traj_step_in_windows = traj_step_in_windows.data(), traj_times, window_size] __device__(int i) {
			if (i % max_traj_len == 0 || traj_times[i] == 0.f)
			{
				traj_step_in_windows[i] = 0;
				return;
			}

			float slice_begin = traj_times[i - 1];
			float slice_end = traj_times[i];

			int wnd_idx = floorf(slice_begin / window_size);
			float wnd_start = wnd_idx * window_size;

			int windows = 0;

			while (slice_end > slice_begin)
			{
				windows++;
				wnd_start = (wnd_idx + windows) * window_size;
				slice_begin = fminf(slice_end, wnd_start);
			}

			traj_step_in_windows[i] = windows;
		});

	size_t total_slices = thrust::reduce(traj_step_in_windows.begin(), traj_step_in_windows.end());

	thrust::exclusive_scan(traj_step_in_windows.begin(), traj_step_in_windows.end() - 1, traj_step_in_windows.begin());
	traj_step_in_windows.back() = total_slices;

	return traj_step_in_windows;
}

void partition(float window_size, thrust::device_ptr<state_t> traj_states, thrust::device_ptr<float> traj_times,
			   thrust::device_ptr<float> traj_tr_entropies, int max_traj_len, int n_trajectories,
			   const thrust::device_vector<int>& traj_step_in_windows, size_t batch_begin_idx, size_t batch_end_idx,
			   thrust::device_vector<state_t>& traj_states_divided, thrust::device_vector<float>& traj_slices_divided,
			   thrust::device_vector<int>& traj_slice_window_divided,
			   thrust::device_vector<float>& traj_tr_entropies_divided)
{
	size_t batch_begin = traj_step_in_windows[batch_begin_idx];
	size_t batch_end = traj_step_in_windows[batch_end_idx];

	size_t batch_size = batch_end - batch_begin;

	traj_states_divided.resize(batch_size);
	traj_slices_divided.resize(batch_size);
	traj_slice_window_divided.resize(batch_size);
	traj_tr_entropies_divided.resize(batch_size);

	thrust::for_each(
		thrust::device, thrust::make_counting_iterator(batch_begin_idx), thrust::make_counting_iterator(batch_end_idx),
		[traj_states, traj_times, traj_tr_entropies, traj_states_divided = traj_states_divided.data(),
		 traj_slices_divided = traj_slices_divided.data(), traj_slice_window_divided = traj_slice_window_divided.data(),
		 traj_tr_entropies_divided = traj_tr_entropies_divided.data(), max_traj_len, window_size,
		 traj_step_in_windows = traj_step_in_windows.data(), batch_begin] __device__(int i) {
			if (i % max_traj_len == 0 || traj_times[i] == 0.f)
			{
				return;
			}

			size_t offset = traj_step_in_windows[i] - batch_begin;

			state_t state = traj_states[i];
			float tr_h = traj_tr_entropies[i];

			float slice_begin = traj_times[i - 1];
			float slice_end = traj_times[i];

			int wnd_idx = floorf(slice_begin / window_size);

			while (slice_end > slice_begin)
			{
				float wnd_end = (wnd_idx + 1) * window_size;
				float slice_in_wnd = fminf(slice_end, wnd_end) - slice_begin;

				traj_states_divided[offset] = state;
				traj_tr_entropies_divided[offset] = tr_h;
				traj_slices_divided[offset] = slice_in_wnd;
				traj_slice_window_divided[offset] = wnd_idx;

				offset++;
				wnd_idx++;

				slice_begin = fminf(slice_end, wnd_end);
			}
		});
}

void window_average_stats::process_batch(float window_size, float max_time, state_t internal_mask,
										 thrust::device_ptr<state_t> traj_states, thrust::device_ptr<float> traj_times,
										 thrust::device_ptr<float> traj_tr_entropies, int max_traj_len,
										 int n_trajectories)
{
	size_t windows_count = std::ceil(max_time / window_size);

	timer t;
	long long transform_time = 0.f, count_time = 0.f, partition_time = 0.f, sort_time = 0.f, reduce_time = 0.f,
			  update_time = 0.f;

	t.start();

	thrust::transform(traj_states, traj_states + n_trajectories * max_traj_len, traj_states,
					  [internal_mask] __device__(state_t s) { return s & ~internal_mask; });

	t.stop();

	transform_time = t.millisecs();

	// host and device result arrays
	thrust::device_vector<int> d_res_window_idxs;
	thrust::device_vector<state_t> d_res_states;
	thrust::device_vector<float> d_res_times;
	thrust::device_vector<float> d_res_tr_entropies;
	std::vector<int> h_res_window_idxs;
	std::vector<state_t> h_res_states;
	std::vector<float> h_res_times;
	std::vector<float> h_res_tr_entropies;

	t.start();

	thrust::device_vector<int> divided_count =
		partition_count(window_size, traj_states, traj_times, traj_tr_entropies, max_traj_len, n_trajectories);

	// divide windows to batches of batch_size_limit so we do not OOM
	std::vector<size_t> batch_indices;
	{
		batch_indices.push_back(0);
		while (batch_indices.back() != n_trajectories * max_traj_len)
		{
			size_t new_idx = thrust::lower_bound(divided_count.begin(), divided_count.end(),
												 (batch_size_limit_)*batch_indices.size())
							 - divided_count.begin() - 1;

			batch_indices.push_back(new_idx);
		}
	}

	t.stop();

	count_time += t.millisecs();

	thrust::device_vector<state_t> batch_states(batch_size_limit_ + windows_count);
	thrust::device_vector<int> batch_window_idxs(batch_size_limit_ + windows_count);
	thrust::device_vector<float> batch_slices(batch_size_limit_ + windows_count),
		batch_tr_entropies(batch_size_limit_ + windows_count);

	// we compute a batch of windows at a time
	for (int batch_i = 0; batch_i < batch_indices.size() - 1; batch_i++)
	{
		size_t batch_idx_begin = batch_indices[batch_i];
		size_t batch_idx_end = batch_indices[batch_i + 1];

		t.start();

		partition(window_size, traj_states, traj_times, traj_tr_entropies, max_traj_len, n_trajectories, divided_count,
				  batch_idx_begin, batch_idx_end, batch_states, batch_slices, batch_window_idxs, batch_tr_entropies);

		t.stop();

		partition_time += t.millisecs();

		size_t batch_size = batch_states.size();

		t.start();

		// let us sort ((window_idx, state, time_b, time_e)) array by (window_idx, state) key
		// so we can reduce it in the next step
		auto key_begin = thrust::make_zip_iterator(batch_window_idxs.begin(), batch_states.begin());
		auto data_begin = thrust::make_zip_iterator(batch_slices.begin(), batch_tr_entropies.begin());
		thrust::sort_by_key(key_begin, key_begin + batch_size, data_begin);

		t.stop();

		sort_time += t.millisecs();

		// create transform iterator, which computes the transition entropy multiplied by the time slice
		auto weighted_tr_entropy_begin =
			thrust::make_transform_iterator(thrust::make_zip_iterator(batch_slices.begin(), batch_tr_entropies.begin()),
											thrust::make_zip_function(thrust::multiplies<float>()));

		t.start();

		// we compute the size of the result (sum of unique states in each window)
		size_t result_size = thrust::unique_count(key_begin, key_begin + batch_size);

		d_res_window_idxs.resize(result_size);
		d_res_states.resize(result_size);
		d_res_times.resize(result_size);
		d_res_tr_entropies.resize(result_size);

		// reduce sorted array of (state, (time_slice, weighted_tr_entropy))
		// after this we have unique states in the first result array and sum of slices and weighted entropies in the
		// second result
		thrust::reduce_by_key(key_begin, key_begin + batch_size,
							  thrust::make_zip_iterator(batch_slices.begin(), weighted_tr_entropy_begin),
							  thrust::make_zip_iterator(d_res_window_idxs.begin(), d_res_states.begin()),
							  thrust::make_zip_iterator(d_res_times.begin(), d_res_tr_entropies.begin()));

		t.stop();

		reduce_time += t.millisecs();

		t.start();

		h_res_window_idxs.resize(result_size);
		h_res_states.resize(result_size);
		h_res_times.resize(result_size);
		h_res_tr_entropies.resize(result_size);

		// copy result data into host
		CUDA_CHECK(cudaMemcpy(h_res_window_idxs.data(), thrust::raw_pointer_cast(d_res_window_idxs.data()),
							  result_size * sizeof(int), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(h_res_states.data(), thrust::raw_pointer_cast(d_res_states.data()),
							  result_size * sizeof(state_t), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(h_res_times.data(), thrust::raw_pointer_cast(d_res_times.data()),
							  result_size * sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CHECK(cudaMemcpy(h_res_tr_entropies.data(), thrust::raw_pointer_cast(d_res_tr_entropies.data()),
							  result_size * sizeof(float), cudaMemcpyDeviceToHost));

		// update
		result_.resize(windows_count);
		for (size_t i = 0; i < result_size; ++i)
		{
			auto it = result_[h_res_window_idxs[i]].find(h_res_states[i]);

			if (it != result_[h_res_window_idxs[i]].end())
			{
				result_[h_res_window_idxs[i]][h_res_states[i]].first += h_res_times[i];
				result_[h_res_window_idxs[i]][h_res_states[i]].second += h_res_tr_entropies[i];
			}
			else
			{
				result_[h_res_window_idxs[i]][h_res_states[i]] = std::make_pair(h_res_times[i], h_res_tr_entropies[i]);
			}
		}

		t.stop();

		update_time += t.millisecs();

		batch_idx_begin = batch_idx_end;
	}

	if (print_diags)
	{
		std::cout << "window_average> batches count: " << batch_indices.size() - 1 << std::endl;
		std::cout << "window_average> transform_time: " << transform_time << "ms" << std::endl;
		std::cout << "window_average> count_time: " << count_time << "ms" << std::endl;
		std::cout << "window_average> partition_time: " << partition_time << "ms" << std::endl;
		std::cout << "window_average> sort_time: " << sort_time << "ms" << std::endl;
		std::cout << "window_average> reduce_time: " << reduce_time << "ms" << std::endl;
		std::cout << "window_average> update_time: " << update_time << "ms" << std::endl;
	}
}

void window_average_stats::visualize(float window_size, int n_trajectories, const char* const* nodes)
{
	for (size_t i = 0; i < result_.size(); ++i)
	{
		auto w = result_[i];

		float entropy = 0.f;
		float wnd_tr_entropy = 0.f;
		for (const auto& p : w)
		{
			auto prob = p.second.first / (n_trajectories * window_size);
			auto tr_ent = p.second.second / (n_trajectories * window_size);

			entropy += -std::log2(prob) * prob;
			wnd_tr_entropy += tr_ent;
		}

		std::cout << "window (" << i * window_size << ", " << (i + 1) * window_size << "]" << std::endl;
		std::cout << "entropy: " << entropy << std::endl;
		std::cout << "transition entropy: " << wnd_tr_entropy << std::endl;

		for (const auto& p : w)
		{
			auto prob = p.second.first / (n_trajectories * window_size);
			auto tr_entropy = p.second.second / (n_trajectories * window_size);
			auto state = p.first;

			std::cout << prob << " " << to_string(state, nodes) << std::endl;
		}
	}
}
