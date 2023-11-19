// #include <device_launch_parameters.h>

// #include <thrust/binary_search.h>
// #include <thrust/device_free.h>
// #include <thrust/device_malloc.h>
// #include <thrust/device_vector.h>
// #include <thrust/execution_policy.h>
// #include <thrust/sort.h>
// #include <thrust/unique.h>
// #include <thrust/zip_function.h>
// #include <fstream>

// #include "../diagnostics.h"
// #include "../utils.h"
// #include "window_average.h"

// __global__ void into_windows_size(int max_traj_len, int n_trajectories, float window_size,
// 								  const float* __restrict__ traj_times, int* __restrict__ step_window_sizes)
// {
// 	auto id = blockIdx.x * blockDim.x + threadIdx.x;

// 	if (id >= max_traj_len * n_trajectories)
// 		return;

// 	if (id % max_traj_len == 0 || traj_times[id] == 0.f)
// 	{
// 		step_window_sizes[id] = 0;
// 		return;
// 	}

// 	float slice_begin = traj_times[id - 1];
// 	float slice_end = traj_times[id];

// 	int wnd_idx = floorf(slice_begin / window_size);
// 	float wnd_start = wnd_idx * window_size;

// 	int windows = 0;

// 	while (slice_end > slice_begin)
// 	{
// 		windows++;
// 		wnd_start = (wnd_idx + windows) * window_size;
// 		slice_begin = fminf(slice_end, wnd_start);
// 	}

// 	step_window_sizes[id] = windows;
// }

// __global__ void into_windows(int max_traj_len, int batch_end, float window_size, int idx_start, int offset,
// 							 const state_t* __restrict__ traj_states, const float* __restrict__ traj_times,
// 							 const float* __restrict__ traj_tr_entropies, const int* __restrict__ step_window_sizes,
// 							 state_t* __restrict__ windowed_traj_states, float* __restrict__ windowed_traj_slices,
// 							 float* __restrict__ windowed_traj_tr_entropies, int* __restrict__ window_indices)
// {
// 	auto id = blockIdx.x * blockDim.x + threadIdx.x + idx_start;

// 	if (id >= batch_end)
// 		return;

// 	if (id % max_traj_len == 0 || traj_times[id] == 0.f)
// 	{
// 		return;
// 	}

// 	const state_t state = traj_states[id];
// 	const float tr_h = traj_tr_entropies[id];

// 	int out_idx = step_window_sizes[id] - offset;

// 	float slice_begin = traj_times[id - 1];
// 	float slice_end = traj_times[id];
// 	int wnd_idx = floorf(slice_begin / window_size);

// 	while (slice_end > slice_begin)
// 	{
// 		float wnd_end = (wnd_idx + 1) * window_size;
// 		float slice_in_wnd = fminf(slice_end, wnd_end) - slice_begin;

// 		windowed_traj_states[out_idx] = state;
// 		windowed_traj_slices[out_idx] = slice_in_wnd;
// 		windowed_traj_tr_entropies[out_idx] = tr_h;
// 		window_indices[out_idx] = wnd_idx;

// 		out_idx++;
// 		wnd_idx++;

// 		slice_begin = fminf(slice_end, wnd_end);
// 	}
// }

// namespace thrust {
// template <>
// struct plus<tuple<float, float>>
// {
// 	__host__ __device__ tuple<float, float> operator()(const tuple<float, float>& lhs,
// 													   const tuple<float, float>& rhs) const
// 	{
// 		return make_tuple(lhs.get<0>() + rhs.get<0>(), lhs.get<1>() + rhs.get<1>());
// 	}
// };
// } // namespace thrust

// window_average_stats::window_average_stats(float window_size, float max_time, state_t internal_mask,
// 										   size_t max_traj_len, size_t max_n_trajectories)
// 	: window_size_(window_size),
// 	  max_time_(max_time),
// 	  internal_mask_(internal_mask),
// 	  max_traj_len_(max_traj_len),
// 	  max_n_trajectories_(max_n_trajectories),
// 	  batch_size_limit_(50'000'000)
// {
// 	timer t;
// 	t.start();

// 	steps_by_window_sizes_ = thrust::device_malloc<int>(max_n_trajectories * max_traj_len + 1);

// 	size_t windows_count = std::ceil(max_time / window_size);
// 	windowed_traj_states_ = thrust::device_malloc<state_t>(batch_size_limit_);
// 	windowed_traj_slices_ = thrust::device_malloc<float>(batch_size_limit_);
// 	windowed_traj_tr_entropies_ = thrust::device_malloc<float>(batch_size_limit_);
// 	window_indices_ = thrust::device_malloc<int>(batch_size_limit_);

// 	t.stop();

// 	if (print_diags)
// 	{
// 		std::cout << "window_average> init_time: " << t.millisecs() << "ms" << std::endl;
// 	}
// }

// window_average_stats::~window_average_stats()
// {
// 	thrust::device_free(steps_by_window_sizes_);
// 	thrust::device_free(windowed_traj_states_);
// 	thrust::device_free(windowed_traj_slices_);
// 	thrust::device_free(windowed_traj_tr_entropies_);
// 	thrust::device_free(window_indices_);
// }

// void window_average_stats::partition_steps_into_windows_size(thrust::device_ptr<float> traj_times,
// 															 int n_trajectories_batch)
// {
// 	// we divide trajectory steps into multiple steps honoring window sizes
// 	into_windows_size<<<DIV_UP(n_trajectories_batch * max_traj_len_, 256), 256>>>(
// 		max_traj_len_, n_trajectories_batch, window_size_, traj_times.get(), steps_by_window_sizes_.get());

// 	CUDA_CHECK(cudaDeviceSynchronize());

// 	// we count the number of windowed steps
// 	size_t total_steps =
// 		thrust::reduce(steps_by_window_sizes_, steps_by_window_sizes_ + n_trajectories_batch * max_traj_len_);

// 	// we compute prefix sum for offsetting
// 	thrust::exclusive_scan(steps_by_window_sizes_, steps_by_window_sizes_ + n_trajectories_batch * max_traj_len_,
// 						   steps_by_window_sizes_);
// 	steps_by_window_sizes_[n_trajectories_batch * max_traj_len_] = total_steps;
// }

// int window_average_stats::partition_steps_into_windows(thrust::device_ptr<unit_state_t> traj_states,
// 													   thrust::device_ptr<float> traj_times,
// 													   thrust::device_ptr<float> traj_tr_entropies,
// 													   int n_trajectories_batch, int& last_batch_end,
// 													   int& cumul_batch_size)
// {
// 	size_t new_batch_end =
// 		thrust::lower_bound(steps_by_window_sizes_, steps_by_window_sizes_ + n_trajectories_batch * max_traj_len_ + 1,
// 							batch_size_limit_ + cumul_batch_size)
// 		- steps_by_window_sizes_ - 1;

// 	int batch_size = steps_by_window_sizes_[new_batch_end] - cumul_batch_size;

// 	into_windows<<<DIV_UP(new_batch_end - last_batch_end, 256), 256>>>(
// 		max_traj_len_, new_batch_end, window_size_, last_batch_end, cumul_batch_size, traj_states.get(),
// 		traj_times.get(), traj_tr_entropies.get(), steps_by_window_sizes_.get(), windowed_traj_states_.get(),
// 		windowed_traj_slices_.get(), windowed_traj_tr_entropies_.get(), window_indices_.get());

// 	CUDA_CHECK(cudaDeviceSynchronize());

// 	last_batch_end = new_batch_end;
// 	cumul_batch_size += batch_size;

// 	return batch_size;
// }

// void window_average_stats::process_batch(thrust::device_ptr<unit_state_t> traj_states, thrust::device_ptr<float> traj_times,
// 										 thrust::device_ptr<float> traj_tr_entropies, thrust::device_ptr<unit_state_t>,
// 										 thrust::device_ptr<trajectory_status>, int n_trajectories)
// {
// 	process_batch_internal(traj_states, traj_times, traj_tr_entropies, n_trajectories);
// }

// void window_average_stats::process_batch_internal(thrust::device_ptr<unit_state_t> traj_states,
// 												  thrust::device_ptr<float> traj_times,
// 												  thrust::device_ptr<float> traj_tr_entropies, int n_trajectories)
// {
// 	size_t windows_count = std::ceil(max_time_ / window_size_);

// 	timer t;
// 	long long transform_time = 0.f, count_time = 0.f, partition_time = 0.f, sort_time = 0.f, reduce_time = 0.f,
// 			  update_time = 0.f;

// 	t.start();

// 	thrust::transform(traj_states, traj_states + n_trajectories * max_traj_len_, traj_states,
// 					  [internal_mask = internal_mask_] __device__(state_t s) { return s & ~internal_mask; });

// 	t.stop();

// 	transform_time = t.millisecs();

// 	// host and device result arrays
// 	thrust::device_vector<int> d_res_window_idxs;
// 	thrust::device_vector<state_t> d_res_states;
// 	thrust::device_vector<float> d_res_times;
// 	thrust::device_vector<float> d_res_tr_entropies;
// 	std::vector<int> h_res_window_idxs;
// 	std::vector<state_t> h_res_states;
// 	std::vector<float> h_res_times;
// 	std::vector<float> h_res_tr_entropies;

// 	t.start();

// 	partition_steps_into_windows_size(traj_times, n_trajectories);

// 	t.stop();

// 	count_time += t.millisecs();

// 	// we compute by batches
// 	int last_batch_end = 0;
// 	int cumul_batch_size = 0;
// 	int iterations = 0;
// 	while (last_batch_end != n_trajectories * max_traj_len_)
// 	{
// 		t.start();

// 		size_t batch_size = partition_steps_into_windows(traj_states, traj_times, traj_tr_entropies, n_trajectories,
// 														 last_batch_end, cumul_batch_size);

// 		t.stop();

// 		partition_time += t.millisecs();

// 		t.start();

// 		// let us sort ((window_idx, state, time_b, time_e)) array by (window_idx, state) key
// 		// so we can reduce it in the next step
// 		auto key_begin = thrust::make_zip_iterator(window_indices_, windowed_traj_states_);
// 		auto data_begin = thrust::make_zip_iterator(windowed_traj_slices_, windowed_traj_tr_entropies_);
// 		thrust::sort_by_key(key_begin, key_begin + batch_size, data_begin);

// 		t.stop();

// 		sort_time += t.millisecs();

// 		// create transform iterator, which computes the transition entropy multiplied by the time slice
// 		auto weighted_tr_entropy_begin =
// 			thrust::make_transform_iterator(data_begin, thrust::make_zip_function(thrust::multiplies<float>()));

// 		t.start();

// 		// we compute the size of the result (sum of unique states in each window)
// 		size_t result_size = thrust::unique_count(key_begin, key_begin + batch_size);

// 		d_res_window_idxs.resize(result_size);
// 		d_res_states.resize(result_size);
// 		d_res_times.resize(result_size);
// 		d_res_tr_entropies.resize(result_size);

// 		// reduce sorted array of (state, (time_slice, weighted_tr_entropy))
// 		// after this we have unique states in the first result array and sum of slices and weighted entropies in the
// 		// second result
// 		thrust::reduce_by_key(key_begin, key_begin + batch_size,
// 							  thrust::make_zip_iterator(windowed_traj_slices_, weighted_tr_entropy_begin),
// 							  thrust::make_zip_iterator(d_res_window_idxs.begin(), d_res_states.begin()),
// 							  thrust::make_zip_iterator(d_res_times.begin(), d_res_tr_entropies.begin()));

// 		t.stop();

// 		reduce_time += t.millisecs();

// 		t.start();

// 		h_res_window_idxs.resize(result_size);
// 		h_res_states.resize(result_size);
// 		h_res_times.resize(result_size);
// 		h_res_tr_entropies.resize(result_size);

// 		// copy result data into host
// 		CUDA_CHECK(cudaMemcpy(h_res_window_idxs.data(), thrust::raw_pointer_cast(d_res_window_idxs.data()),
// 							  result_size * sizeof(int), cudaMemcpyDeviceToHost));
// 		CUDA_CHECK(cudaMemcpy(h_res_states.data(), thrust::raw_pointer_cast(d_res_states.data()),
// 							  result_size * sizeof(state_t), cudaMemcpyDeviceToHost));
// 		CUDA_CHECK(cudaMemcpy(h_res_times.data(), thrust::raw_pointer_cast(d_res_times.data()),
// 							  result_size * sizeof(float), cudaMemcpyDeviceToHost));
// 		CUDA_CHECK(cudaMemcpy(h_res_tr_entropies.data(), thrust::raw_pointer_cast(d_res_tr_entropies.data()),
// 							  result_size * sizeof(float), cudaMemcpyDeviceToHost));

// 		// update
// 		result_.resize(windows_count);
// 		for (size_t i = 0; i < result_size; ++i)
// 		{
// 			auto it = result_[h_res_window_idxs[i]].find(h_res_states[i]);

// 			if (it != result_[h_res_window_idxs[i]].end())
// 			{
// 				result_[h_res_window_idxs[i]][h_res_states[i]].first += h_res_times[i];
// 				result_[h_res_window_idxs[i]][h_res_states[i]].second += h_res_tr_entropies[i];
// 			}
// 			else
// 			{
// 				result_[h_res_window_idxs[i]][h_res_states[i]] = std::make_pair(h_res_times[i], h_res_tr_entropies[i]);
// 			}
// 		}

// 		t.stop();

// 		update_time += t.millisecs();

// 		iterations++;
// 	}

// 	if (print_diags)
// 	{
// 		std::cout << "window_average> iterations: " << iterations << std::endl;
// 		std::cout << "window_average> transform_time: " << transform_time << "ms" << std::endl;
// 		std::cout << "window_average> count_time: " << count_time << "ms" << std::endl;
// 		std::cout << "window_average> partition_time: " << partition_time << "ms" << std::endl;
// 		std::cout << "window_average> sort_time: " << sort_time << "ms" << std::endl;
// 		std::cout << "window_average> reduce_time: " << reduce_time << "ms" << std::endl;
// 		std::cout << "window_average> update_time: " << update_time << "ms" << std::endl;
// 	}
// }

// void window_average_stats::visualize(int n_trajectories, const std::vector<std::string>& nodes)
// {
// 	for (size_t i = 0; i < result_.size(); ++i)
// 	{
// 		auto w = result_[i];

// 		float entropy = 0.f;
// 		float wnd_tr_entropy = 0.f;
// 		for (const auto& p : w)
// 		{
// 			auto prob = p.second.first / (n_trajectories * window_size_);
// 			auto tr_ent = p.second.second / (n_trajectories * window_size_);

// 			entropy += -std::log2(prob) * prob;
// 			wnd_tr_entropy += tr_ent;
// 		}

// 		std::cout << "window (" << i * window_size_ << ", " << (i + 1) * window_size_ << "]" << std::endl;
// 		std::cout << "entropy: " << entropy << std::endl;
// 		std::cout << "transition entropy: " << wnd_tr_entropy << std::endl;

// 		for (const auto& p : w)
// 		{
// 			auto prob = p.second.first / (n_trajectories * window_size_);
// 			auto tr_entropy = p.second.second / (n_trajectories * window_size_);
// 			auto state = p.first;

// 			std::cout << prob << " " << to_string(state, nodes) << std::endl;
// 		}
// 	}
// }

// void window_average_stats::write_csv(int n_trajectories, const std::vector<std::string>& nodes,
// 									 const std::string prefix)
// {
// 	std::ofstream ofs;

// 	ofs.open(prefix + "_probtraj.csv");
// 	if (ofs)
// 	{
// 		// Computing max states for header
// 		size_t max_states = 0;
// 		for (size_t i = 0; i < result_.size(); ++i)
// 		{
// 			max_states = std::max(max_states, result_[i].size());
// 		}

// 		// Writing header
// 		ofs << "Time\tTH\tErrorTH\tH\tHD=0";
// 		for (size_t i = 0; i < max_states; i++)
// 		{
// 			ofs << "\tState\tProba\tErrorProba";
// 		}
// 		ofs << std::endl;

// 		// Writing trajectories
// 		for (size_t i = 0; i < result_.size(); ++i)
// 		{
// 			auto w = result_[i];

// 			float entropy = 0.f;
// 			float wnd_tr_entropy = 0.f;
// 			for (const auto& p : w)
// 			{
// 				auto prob = p.second.first / (n_trajectories * window_size_);
// 				auto tr_ent = p.second.second / (n_trajectories * window_size_);

// 				entropy += -std::log2(prob) * prob;
// 				wnd_tr_entropy += tr_ent;
// 			}
// 			ofs << i * window_size_ << "\t";
// 			ofs << wnd_tr_entropy << "\t" << 0.f << "\t" << entropy << "\t" << 0.f;

// 			for (const auto& p : w)
// 			{
// 				auto prob = p.second.first / (n_trajectories * window_size_);
// 				auto tr_entropy = p.second.second / (n_trajectories * window_size_);
// 				auto state = p.first;

// 				ofs << "\t" << to_string(state, nodes) << "\t" << prob << "\t" << 0.f;
// 			}
// 			ofs << std::endl;
// 		}
// 	}
// }
