#include <fstream>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "../diagnostics.h"
#include "fixed_states.h"

template <int state_words>
void fixed_states_stats<state_words>::process_batch_internal(
	thrust::device_ptr<static_state_t<state_words>> last_states, thrust::device_ptr<trajectory_status> traj_statuses,
	int n_trajectories)
{
	timer t;
	float copy_sort_reduce_time = 0.f, update_time = 0.f;

	t.start();

	auto fp_pred = [] __device__(trajectory_status t) { return t == trajectory_status::FIXED_POINT; };

	size_t finished_trajs_size = thrust::count_if(traj_statuses, traj_statuses + n_trajectories, fp_pred);

	if (finished_trajs_size == 0)
		return;

	thrust::device_vector<static_state_t<state_words>> final_states(finished_trajs_size);

	thrust::copy_if(last_states, last_states + n_trajectories, traj_statuses, final_states.begin(), fp_pred);

	thrust::sort(final_states.begin(), final_states.end());

	size_t unique_fixed_points_size = thrust::unique_count(final_states.begin(), final_states.end());

	thrust::device_vector<static_state_t<state_words>> unique_fixed_points(unique_fixed_points_size);
	thrust::device_vector<int> unique_fixed_points_count(unique_fixed_points_size);

	thrust::reduce_by_key(final_states.begin(), final_states.end(), thrust::make_constant_iterator(1),
						  unique_fixed_points.begin(), unique_fixed_points_count.begin());

	t.stop();
	copy_sort_reduce_time = t.millisecs();
	t.start();

	std::vector<static_state_t<state_words>> h_unique_fixed_points(unique_fixed_points_size);
	std::vector<int> h_unique_fixed_points_count(unique_fixed_points_size);

	thrust::copy(unique_fixed_points.begin(), unique_fixed_points.end(), h_unique_fixed_points.begin());
	thrust::copy(unique_fixed_points_count.begin(), unique_fixed_points_count.end(),
				 h_unique_fixed_points_count.begin());

	for (size_t i = 0; i < unique_fixed_points_size; i++)
	{
		auto it = result_.find(h_unique_fixed_points[i]);

		if (it != result_.end())
			result_[h_unique_fixed_points[i]] += h_unique_fixed_points_count[i];
		else
			result_[h_unique_fixed_points[i]] = h_unique_fixed_points_count[i];
	}

	t.stop();

	update_time = t.millisecs();

	if (print_diags)
	{
		std::cout << "fixed_states> copy_sort_reduce_time: " << copy_sort_reduce_time << "ms" << std::endl;
		std::cout << "fixed_states> update_time: " << update_time << "ms" << std::endl;
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
	std::cout << "fixed points:" << std::endl;

	for (const auto& p : result_)
	{
		std::cout << (float)p.second / (float)n_trajectories << " "
				  << state_t(nodes.size(), p.first.data).to_string(nodes) << std::endl;
	}
}

template <int state_words>
void fixed_states_stats<state_words>::write_csv(int n_trajectories, const std::vector<std::string>& nodes,
												const std::string prefix)
{
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
