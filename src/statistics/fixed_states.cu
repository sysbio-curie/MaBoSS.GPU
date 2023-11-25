#include <fstream>

#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

#include "../timer.h"
#include "../utils.h"
#include "fixed_states.h"

struct select_ftor
{
	__device__ __forceinline__ bool operator()(trajectory_status t) const
	{
		return t == trajectory_status::FIXED_POINT;
	}
};

template <int state_words>
struct compare_ftor
{
	__device__ __forceinline__ bool operator()(const static_state_t<state_words>& lhs,
											   const static_state_t<state_words>& rhs) const
	{
		for (int i = state_words - 1; i >= 0; i--)
			if (lhs.data[i] != rhs.data[i])
				return lhs.data[i] < rhs.data[i];
		return false;
	}
};

template <int state_words>
void fixed_states_stats<state_words>::initialize_temp_storage(
	thrust::device_ptr<static_state_t<state_words>> last_states, thrust::device_ptr<trajectory_status> traj_statuses,
	int n_trajectories)
{
	if (d_tmp_storage_ != nullptr)
	{
		return;
	}

	timer_stats stats("fixed_states_stats> initialize");

	size_t temp_storage_bytes_if = 0;
	cub::DeviceSelect::Flagged(
		d_tmp_storage_, tmp_storage_bytes_, last_states.get(),
		cub::TransformInputIterator<bool, select_ftor, trajectory_status*>(traj_statuses.get(), select_ftor()),
		d_fixed_copy_, d_out_num, n_trajectories);

	std::size_t temp_storage_bytes_sort = 0;
	cub::DeviceMergeSort::SortKeys(d_tmp_storage_, temp_storage_bytes_sort, d_fixed_copy_, n_trajectories,
								   compare_ftor<state_words>());

	size_t temp_storage_bytes_encode = 0;
	cub::DeviceRunLengthEncode::Encode(d_tmp_storage_, tmp_storage_bytes_, d_fixed_copy_, d_unique_states_,
									   d_unique_states_count_, d_out_num, n_trajectories);

	tmp_storage_bytes_ = std::max(std::max(temp_storage_bytes_if, temp_storage_bytes_encode), temp_storage_bytes_sort);

	CUDA_CHECK(cudaMalloc(&d_tmp_storage_, tmp_storage_bytes_));
	CUDA_CHECK(cudaMalloc(&d_out_num, sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_fixed_copy_, n_trajectories * sizeof(static_state_t<state_words>)));
	CUDA_CHECK(cudaMalloc(&d_unique_states_, n_trajectories * sizeof(static_state_t<state_words>)));
	CUDA_CHECK(cudaMalloc(&d_unique_states_count_, n_trajectories * sizeof(int)));
}

template <int state_words>
void fixed_states_stats<state_words>::process_batch_internal(
	thrust::device_ptr<static_state_t<state_words>> last_states, thrust::device_ptr<trajectory_status> traj_statuses,
	int n_trajectories)
{
	initialize_temp_storage(last_states, traj_statuses, n_trajectories);

	timer_stats stats("fixed_states_stats> process_batch");

	cub::DeviceSelect::Flagged(
		d_tmp_storage_, tmp_storage_bytes_, last_states.get(),
		cub::TransformInputIterator<bool, select_ftor, trajectory_status*>(traj_statuses.get(), select_ftor()),
		d_fixed_copy_, d_out_num, n_trajectories);

	int fixed_count = 0;
	CUDA_CHECK(cudaMemcpy(&fixed_count, d_out_num, sizeof(int), cudaMemcpyDeviceToHost));

	cub::DeviceMergeSort::SortKeys(d_tmp_storage_, tmp_storage_bytes_, d_fixed_copy_, fixed_count,
								   compare_ftor<state_words>());

	cub::DeviceRunLengthEncode::Encode(d_tmp_storage_, tmp_storage_bytes_, d_fixed_copy_, d_unique_states_,
									   d_unique_states_count_, d_out_num, fixed_count);

	int unique_fixed_points_size = 0;
	CUDA_CHECK(cudaMemcpy(&unique_fixed_points_size, d_out_num, sizeof(int), cudaMemcpyDeviceToHost));

	std::vector<static_state_t<state_words>> h_unique_fixed_points(unique_fixed_points_size);
	std::vector<int> h_unique_fixed_points_count(unique_fixed_points_size);

	CUDA_CHECK(cudaMemcpy(h_unique_fixed_points.data(), d_unique_states_,
						  unique_fixed_points_size * sizeof(static_state_t<state_words>), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_unique_fixed_points_count.data(), d_unique_states_count_,
						  unique_fixed_points_size * sizeof(int), cudaMemcpyDeviceToHost));

	for (size_t i = 0; i < unique_fixed_points_size; i++)
	{
		auto it = result_.find(h_unique_fixed_points[i]);

		if (it != result_.end())
			result_[h_unique_fixed_points[i]] += h_unique_fixed_points_count[i];
		else
			result_[h_unique_fixed_points[i]] = h_unique_fixed_points_count[i];
	}
}

template <int state_words>
void fixed_states_stats<state_words>::process_batch(thrust::device_ptr<state_word_t> traj_states,
													thrust::device_ptr<float> traj_times,
													thrust::device_ptr<float> traj_tr_entropies,
													thrust::device_ptr<state_word_t> last_states,
													thrust::device_ptr<trajectory_status> traj_statuses,
													int n_trajectories)
{
	process_batch_internal(
		thrust::device_pointer_cast<static_state_t<state_words>>((static_state_t<state_words>*)last_states.get()),
		traj_statuses, n_trajectories);
}

template <int state_words>
void fixed_states_stats<state_words>::visualize(int n_trajectories, const std::vector<std::string>& nodes)
{
	timer_stats stats("fixed_states_stats> visualize");

	std::cout << "fixed points:" << std::endl;

	for (const auto& p : result_)
	{
		std::cout << (float)p.second / (float)n_trajectories << " "
				  << state_t(nodes.size(), p.first.data).to_string(nodes) << std::endl;
	}
}

template <int state_words>
void fixed_states_stats<state_words>::write_csv(int n_trajectories, const std::vector<std::string>& nodes,
												const std::string& prefix)
{
	timer_stats stats("fixed_states_stats> write_csv");

	std::ofstream ofs;

	ofs.open(prefix + "_fp.csv");
	if (ofs)
	{
		ofs << "Fixed Points (" << result_.size() << ")" << std::endl;
		ofs << "FP\tProba\tState";

		for (auto& node : nodes)
		{
			ofs << "\t" << node;
		}
		ofs << std::endl;

		int i_fp = 0;
		for (const auto& p : result_)
		{
			state_t runtime_state(nodes.size(), p.first.data);
			ofs << "#" << i_fp << "\t" << ((float)p.second) / n_trajectories << "\t" << runtime_state.to_string(nodes);
			for (int i = 0; i < nodes.size(); i++)
			{
				ofs << "\t" << runtime_state.is_set(i);
			}
			ofs << std::endl;
			i_fp++;
		}
	}
}

template class fixed_states_stats<1>;
template class fixed_states_stats<2>;
template class fixed_states_stats<3>;
template class fixed_states_stats<4>;
template class fixed_states_stats<5>;
template class fixed_states_stats<6>;
template class fixed_states_stats<7>;
template class fixed_states_stats<8>;
